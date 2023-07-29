#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import sys
sys.path.append('..')
import math
import os
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence
import time
import torch
import transformers
import utils
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import Trainer, DataCollatorForLanguageModeling, AutoConfig
from transformers import get_cosine_schedule_with_warmup, LlamaForCausalLM
import colossalai
from colossalai.nn.optimizer import HybridAdam
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.tensor import ProcessGroup, ShardSpec
from colossalai.utils import get_current_device, print_rank_0
from colossalai.zero import ColoInitContext, GeminiAdamOptimizer, zero_optim_wrapper
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin, TorchFSDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.pipeline.pipelinable import PipelinableContext
import torch.distributed as dist
from tqdm import tqdm
from colossalai.tensor import ProcessGroup, ShardSpec, ComputePattern, ComputeSpec
from colossalai.nn.parallel.layers import init_colo_module
from colossalai.context import ParallelMode
from colossalai.core import global_context as gpc
from statistics import mean
import GPUtil
import psutil


# from titans.model.vit.vit import _create_vit_model

# from titans.utils import barrier_context


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator, output_dir, tp_degree):
    print_rank_0('[3]Used GPU mem: {0}'.format(
        GPUtil.getGPUs()[0].memoryUsed))  # 32077 MB
    print_rank_0('[3]Virtual used mem: {0}'.format(
        get_size(psutil.virtual_memory().used)))  # 18.82 GB
    torch.cuda.synchronize()
    model.train()
    losses = []
    steps = 0
    with tqdm(dataloader, desc=f'Epoch [{epoch + 1}]', disable=not coordinator.is_master()) as pbar:
        # Iters - 2 GPUs: 13000(=52000/(2*2)), 4 GPUs: 3250(=52000/(4*4)), 8 GPUs: 812
        for batch in pbar:
            # Forward
            # optimizer.zero_grad()
            # print(dist.get_rank())

            batch = move_to_cuda(batch, torch.cuda.current_device())
            outputs = model(use_cache=False, **batch)
            # print_rank_0("cccccccccc")
            # sys.exit()
            loss = outputs['loss']
            # Backward
            booster.backward(loss, optimizer)
            optimizer.step()
            lr_scheduler.step()
            # Print batch loss
            pbar.set_postfix(
                {'loss': loss.item(), 'Memory usage': GPUtil.getGPUs()[0].memoryUsed})
            losses.append(loss.item())
            steps += 1
            if steps % 500 == 0:
                booster.save_model(model,  output_dir, tp_degree=tp_degree)

    print_rank_0('Average loss of epoch {0}: {1:.2f}, Memory usage: {2}'.format(
        epoch + 1, mean(losses), GPUtil.getGPUs()[0].memoryUsed))


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={
                           "help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    do_eval: bool = field(default=False)
    do_train: bool = field(default=True)


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format- KB, MB, GB, TB and PB
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def train():
    tp_degree = 8
    dp_degree = 1
    # 0: by row (bs=8, peak_mem=28487 MB), -1: by col (bs=8, peak_mem=24855 MB)
    dims = 0
    kv_h = 8
    # Launch ColossalAI
    # colossalai.launch_from_torch(config={}, seed=0)
    colossalai.launch_from_torch(config=dict(parallel=dict(data=dp_degree, pipeline=1,
                                                           tensor=dict(size=tp_degree, mode='1d'))))
    coordinator = DistCoordinator()
    world_size = coordinator.world_size
    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # with PipelinableContext():

    shard_pg = ProcessGroup(tp_degree=tp_degree)
    default_dist_spec = ShardSpec([dims], [tp_degree])

    with ColoInitContext(device=get_current_device(), default_dist_spec=default_dist_spec, default_pg=shard_pg):
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path)

        model_config.update({"kv_h": kv_h})
        # model_config.save_pretrained("gqa_llama")
        if 'llama-7b' in model_args.model_name_or_path:
            model = LlamaForCausalLM(model_config)
        elif 'llama-65b' in model_args.model_name_or_path:
            model = LlamaForCausalLM(model_config)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Param: {:.2f}".format(
            tp_degree * pytorch_total_params/1000/1000))
        model.gradient_checkpointing_enable()

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        special_tokens_dict = dict()
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            # "[PAD]" -> only this one is included
            special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        if tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN    # "</s>"
        if tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN    # "<s>"
        if tokenizer.unk_token is None:
            special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN    # "<unk>"

        # smart_tokenizer_and_embedding_resize(
        #     special_tokens_dict=special_tokens_dict,
        #     tokenizer=tokenizer,
        #     model=model,
        # )
        # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
        data_module = load_dataset(data_args.data_path)
        data_module["train"] = load_dataset(data_args.data_path, split="train[:1%]")
        data_module = utils.dataset_mapping(tokenizer, data_module, max_seq_length=2048)
        # 27189 MB // When sharding in with ColoInitContext 5525 MB
        print_rank_0('[1]Used GPU mem: {0}'.format(
            GPUtil.getGPUs()[0].memoryUsed))
        print_rank_0('[1]Virtual used mem: {0}'.format(
            get_size(psutil.virtual_memory().used)))  # 14.61 GB

    from safetensors.torch import load_file
    pretrained_state = {}
    pretrained_state.update(load_file(
        "/home/ubuntu/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/model-00001-of-00002.safetensors"))
    pretrained_state.update(load_file(
        "/home/ubuntu/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/model-00002-of-00002.safetensors"))

    # for n, p in pretrained_state.items():
    #     with open("1.txt", "a") as f:
    #         f.writelines(str(n) + " " + str(p.shape) + "\n")
    for n, p in model.named_parameters():
        with open(f"row_gqa_{model_config.kv_h}.txt", "a") as f:
            # print(list(pretrained_state.keys())[i], list(pretrained_state.values())[i].shape)
            f.writelines(str(n) + " " + str(p.shape) + "\n")
    # sys.exit()

    for n, p in model.named_parameters():
        x = pretrained_state[n]
        # print(n)
        if not 'norm' in n:
            x = x.chunk(tp_degree, dim=dims)
            x = x[dist.get_rank() % tp_degree]
            q_per_group = model_config.num_attention_heads // model_config.kv_h
            if 'k_proj' in n or 'v_proj' in n:
                x = utils.group_weight(x, x.shape[0], q_per_group)
        # if (x.shape != p.shape):
        #     print(n, x.shape, p.shape)
        p.data.copy_(x)
    # sys.exit()

    ####################################################################
    compute_spec = ComputeSpec(ComputePattern.TP1D)
    init_colo_module(model, compute_spec, pg=shard_pg, recursive=True)

    # Set plugin
    booster_kwargs = {}
    # booster_kwargs['mixed_precision'] = 'fp16'
    plugin = GeminiPlugin(device=get_current_device(),
                          placement_policy='cuda',
                          precision='fp16',
                          pin_memory=False,  # True,
                          strict_ddp_mode=False,
                          initial_scale=2**5)
    # plugin = TorchFSDPPlugin() # does not work with
    # plugin = LowLevelZeroPlugin(stage=2) # All parameters should be in a same process group

    config = {
        'batch_size': training_args.per_device_train_batch_size,
        'lr': training_args.learning_rate,
        'epochs': int(training_args.num_train_epochs),
        'warmup_ratio': training_args.warmup_ratio,
        'weight_decay': training_args.weight_decay,
    }
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)
    dataloader = plugin.prepare_dataloader(data_module['train'],
                                           batch_size=config['batch_size'],
                                           shuffle=True,
                                           drop_last=True,
                                           collate_fn=data_collator)

    # val_dataloader = plugin.prepare_dataloader(data_module['validation'],
    #                                        batch_size=config['batch_size'],
    #                                        shuffle=True,
    #                                        drop_last=True,
    #                                        collate_fn=data_collator)
    # # Set lr scheduler
    total_steps = len(dataloader) * config['epochs']
    num_warmup_steps = int(config['warmup_ratio'] * total_steps)

    ############################ Case 1. Use Booster ############################
    # Set optimizer
    optimizer = HybridAdam(model.parameters(),
                           lr=(config['lr'] * world_size),
                           weight_decay=0.0)

    # Set lr scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=len(dataloader) * config['epochs']
    )

    booster = Booster(plugin=plugin, **booster_kwargs)
    model, optimizer, _, _, _ = booster.boost(model, optimizer)
    # for p in model.unwrap().module.parameters():
    #     print(p.data)
    #     print_rank_0(p.data.size())
    #     sys.exit()

    # Start finetuning
    if training_args.do_train:
        logger.info(f"Start finetuning", ranks=[0])
        output_dir = f'/home/ubuntu/GQA/trained/hellaswag_gqa_{model_config.kv_h}_shard_colo_llama-7b/' + str(
            dist.get_rank()) + '.pt'
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        start = time.time()
        for epoch in range(config['epochs']):
            train_epoch(epoch, model, optimizer, lr_scheduler,
                        dataloader, booster, coordinator, output_dir, tp_degree)
        logger.info(f"Finish finetuning, time:{time.time()-start}", ranks=[0])

        booster.save_model(model,  output_dir, tp_degree=tp_degree)

    if training_args.do_eval:
        with ColoInitContext(device=get_current_device(), default_dist_spec=default_dist_spec, default_pg=shard_pg):
            model_config = AutoConfig.from_pretrained(
                model_args.model_name_or_path)
            model_config.update({"kv_h": 32})
            if 'llama-7b' in model_args.model_name_or_path:
                model = LlamaForCausalLM(model_config)

            model.cuda()
            model.eval()

            tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_args.model_name_or_path,
                use_fast=False,
                model_max_length=512,
            )

            if tokenizer.pad_token is None:
                # For size matching of Colossal-AI
                tokenizer.pad_token = tokenizer.eos_token

            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )
        compute_spec = ComputeSpec(ComputePattern.TP1D)
        init_colo_module(model, compute_spec, pg=shard_pg, recursive=True)
        state_dict = torch.load(
            "/home/ubuntu/GQA/trained/test_gqa_32_shard_colo_llama-7b/" + str(dist.get_rank()) + '.pt')
        model.load_state_dict(state_dict)

        training_args = TrainingArguments(
            output_dir="test_group_{}_checkpoint".format(model_config.kv_h),
            evaluation_strategy="steps",
            weight_decay=0.01,
            # bf16=True,
            fp16=True,
            do_eval=False,
            per_device_train_batch_size=2,
            # learning_rate=learning_rate,
            max_steps=500,
            seed=0,
            push_to_hub=False,
            adam_beta2=0.98,
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            # train_dataset=tokenized_datasets["train"],
            eval_dataset=data_module["test"],
            data_collator=data_collator,
        )

        # trainer.save_model("roberta_hf")
        eval_results = trainer.evaluate()
        print(
            f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}, samples_per_sec: {eval_results['eval_samples_per_second']}")


if __name__ == "__main__":
    train()
