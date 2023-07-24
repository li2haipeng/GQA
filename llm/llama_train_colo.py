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
import psutil
import GPUtil
from statistics import mean
from colossalai.core import global_context as gpc
from colossalai.context import ParallelMode
from colossalai.nn.parallel.layers import init_colo_module
from colossalai.tensor import ProcessGroup, ShardSpec, ComputePattern, ComputeSpec
from tqdm import tqdm
import torch.distributed as dist
from colossalai.pipeline.pipelinable import PipelinableContext
from colossalai.cluster import DistCoordinator
from colossalai.booster.plugin import GeminiPlugin, LowLevelZeroPlugin, TorchDDPPlugin, TorchFSDPPlugin
from colossalai.booster import Booster
from colossalai.zero import ColoInitContext, GeminiAdamOptimizer, zero_optim_wrapper
from colossalai.utils import get_current_device, print_rank_0
from colossalai.tensor import ProcessGroup, ShardSpec
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
import colossalai
from transformers import get_cosine_schedule_with_warmup
from transformers import Trainer, DataCollatorForLanguageModeling, AutoConfig
from datasets import load_dataset
from torch.utils.data import Dataset
import utils
import transformers
import torch
import time
from typing import Dict, Optional, Sequence
from dataclasses import dataclass, field
import logging
import copy
import os


# from titans.model.vit.vit import _create_vit_model

# from titans.utils import barrier_context


def move_to_cuda(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def train_epoch(epoch, model, optimizer, lr_scheduler, dataloader, booster, coordinator):
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
            steps+=1
            if steps >= 50:
                break

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


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(
        special_tokens_dict)  # special_tokens_dict = {'pad_token': '[PAD]'}
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-
                                                num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-
                                                  num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get(
                "input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(
        tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


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
    # for LLaMA models
    import transformers.models.llama.modeling_llama
    # Launch ColossalAI
    # colossalai.launch_from_torch(config={}, seed=0)
    colossalai.launch_from_torch(config=dict(parallel=dict(
        data=1, pipeline=1, tensor=dict(size=tp_degree, mode='1d'))))
    coordinator = DistCoordinator()
    world_size = coordinator.world_size
    shard_pg = ProcessGroup(tp_degree=tp_degree)
    # Manage loggers
    disable_existing_loggers()
    logger = get_dist_logger()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # with PipelinableContext():
    default_dist_spec = ShardSpec([-1], [tp_degree])
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    model_config.update({"kv_h": 8})
    model_config.save_pretrained("gqa_llama")

    with ColoInitContext(device=get_current_device(), default_dist_spec=default_dist_spec, default_pg=shard_pg):
        # with ColoInitContext(device=get_current_device()):
        # from transformers import LlamaConfig
        # configuration = LlamaConfig(vocab_size=31999)
        print_rank_0('[0]Used GPU mem: {0}'.format(
            GPUtil.getGPUs()[0].memoryUsed))  # 1421 MB
        print_rank_0('[0]Virtual total mem: {0}'.format(
            get_size(psutil.virtual_memory().total)))  # 1.10 TB
        print_rank_0('[0]Virtual used mem: {0}'.format(
            get_size(psutil.virtual_memory().used)))  # 14.55 GB
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        #     model_args.model_name_or_path,
        #     cache_dir=training_args.cache_dir,
        #     ignore_mismatched_sizes=True,
        # ) #.to('cuda')

        model = transformers.AutoModelForCausalLM.from_config(model_config)
        model.gradient_checkpointing_enable()
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print("Param: {:.2f}".format(
            tp_degree * pytorch_total_params/1000/1000))

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
        data_module["train"] = load_dataset(
            data_args.data_path, split="train[:1%]")
        data_module = utils.dataset_mapping(
            tokenizer, data_module, max_seq_length=2048)
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
    # for n, p in model.named_parameters():
    #     with open("row_gqa_4.txt", "a") as f:
    #     # print(list(pretrained_state.keys())[i], list(pretrained_state.values())[i].shape)
    #         f.writelines(str(n) + " " + str(p.shape) + "\n")
    # sys.exit()

    for n, p in model.named_parameters():
        x = pretrained_state[n]
        # print(n)
        if not 'norm' in n:
            x = x.chunk(tp_degree, dim=-1)
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
    logger.info(f"Start finetuning", ranks=[0])
    start = time.time()
    for epoch in range(config['epochs']):
        train_epoch(epoch, model, optimizer, lr_scheduler,
                    dataloader, booster, coordinator)
    logger.info(f"Finish finetuning, time:{time.time()-start}", ranks=[0])

    output_dir = f'/home/ubuntu/gqa/trained/gqa_{model_config.kv_h}_shard_colo_llama-7b/' + \
        str(dist.get_rank()) + '.pt'
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    booster.save_model(model,  output_dir, tp_degree=tp_degree)
    # print(output_dir)
    # Finish training and evaluate
    


if __name__ == "__main__":
    train()
