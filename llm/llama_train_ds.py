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
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer, AutoConfig, DataCollatorForLanguageModeling
from datasets import load_dataset

import torch.distributed as dist
import GPUtil
import psutil

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
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

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
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
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
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
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
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

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
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = SupervisedDataset(tokenizer=tokenizer, data_path=data_args.data_path)
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
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # model_args: ModelArguments(model_name_or_path='huggyllama/llama-7b')
    # data_args: DataArguments(data_path='./alpaca_data.json')
    # training_args: TrainingArguments(_n_gpu=1, adafactor=False, adam_beta1=0.9, adam_beta2=0.999,
    # adam_epsilon=1e-08, auto_find_batch_size=False, bf16=True, bf16_full_eval=False, cache_dir=None,
    # data_seed=None, dataloader_drop_last=False, dataloader_num_workers=0, dataloader_pin_memory=True,
    # ddp_backend=None, ddp_bucket_cap_mb=None, ddp_find_unused_parameters=None, ddp_timeout=1800, debug=[],
    # deepspeed=ds_config.json, disable_tqdm=False, do_eval=False, do_predict=False, do_train=False, 
    # eval_accumulation_steps=None, eval_delay=0, eval_steps=None, evaluation_strategy=no, fp16=False,
    # fp16_backend=auto, fp16_full_eval=False, fp16_opt_level=O1, fsdp=[], 
    # fsdp_config={'fsdp_min_num_params': 0, 'xla': False, 'xla_fsdp_grad_ckpt': False}, fsdp_min_num_params=0,
    # fsdp_transformer_layer_cls_to_wrap=None, full_determinism=False, gradient_accumulation_steps=8, 
    # gradient_checkpointing=False, greater_is_better=None, group_by_length=False, half_precision_backend=auto, 
    # hub_model_id=None, hub_private_repo=False, hub_strategy=every_save, hub_token=<HUB_TOKEN>,
    # ignore_data_skip=False, include_inputs_for_metrics=False, jit_mode_eval=False, label_names=None, 
    # label_smoothing_factor=0.0, learning_rate=2e-05, length_column_name=length, load_best_model_at_end=False,
    # local_rank=0, log_level=passive, log_level_replica=warning, log_on_each_node=True,
    # logging_dir=./trained/runs/Jul11_22-07-12_ip-172-31-23-126, logging_first_step=False, 
    # logging_nan_inf_filter=True, logging_steps=1.0, logging_strategy=steps, lr_scheduler_type=cosine,
    # max_grad_norm=1.0, max_steps=-1, metric_for_best_model=None, model_max_length=512, mp_parameters=,
    # no_cuda=False, num_train_epochs=3.0, optim=adamw_torch, optim_args=None, output_dir=./trained, 
    # overwrite_output_dir=False, past_index=-1, per_device_eval_batch_size=4, per_device_train_batch_size=8, 
    # prediction_loss_only=False, push_to_hub=False, push_to_hub_model_id=None, push_to_hub_organization=None,
    # push_to_hub_token=<PUSH_TO_HUB_TOKEN>, ray_scope=last, remove_unused_columns=True, report_to=[], 
    # resume_from_checkpoint=None, run_name=./trained, save_on_each_node=False, save_safetensors=False,
    # save_steps=2000, save_strategy=steps, save_total_limit=1, seed=42, sharded_ddp=[], 
    # skip_memory_metrics=True, (for memory metrics!!)
    # tf32=None, torch_compile=False, torch_compile_backend=None, torch_compile_mode=None, torchdynamo=None, 
    # tpu_metrics_debug=False, tpu_num_cores=None, use_ipex=False, use_legacy_prediction_loop=False,
    # use_mps_device=False, warmup_ratio=0.03, warmup_steps=0, weight_decay=0.0, xpu_backend=None,)
    ######## In the case of 'fsdp' ########
    # fsdp=[<FSDPOption.FULL_SHARD: 'full_shard'>, <FSDPOption.AUTO_WRAP: 'auto_wrap'>],
    # fsdp_config={'fsdp_min_num_params': 0, 'fsdp_transformer_layer_cls_to_wrap': ['LlamaDecoderLayer'], 
    # 'xla': False, 'xla_fsdp_grad_ckpt': False}, fsdp_min_num_params=0, 
    # fsdp_transformer_layer_cls_to_wrap=LlamaDecoderLayer,

    if dist.get_rank() == 0:
        print('[0]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed))    # 3 MB
        print('[0]Virtual total mem: {0}'.format(get_size(psutil.virtual_memory().total))) # 1.10 TB
        print('[0]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 6.20 GB
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    model_config.update({"kv_h": 32})
    model_config.save_pretrained("gqa_llama")
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    # ).to('cuda')
    model = transformers.AutoModelForCausalLM.from_config(model_config)

    model.gradient_checkpointing_enable()   ###

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    if dist.get_rank() == 0:
        print('[1]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed))    # ZeRO-3: 5449 MB, ZeRO-2: 27625 MB
        print('[1]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 5.65 GB

    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    data_module = load_dataset(data_args.data_path)
    data_module["train"] = load_dataset(data_args.data_path, split="train[:1%]")
    data_module = utils.dataset_mapping(tokenizer, data_module, max_seq_length=2048)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # {'train_dataset': <__main__.SupervisedDataset object at 0x7fde04114820>, 'eval_dataset': None, 
    # 'data_collator': DataCollatorForSupervisedDataset(tokenizer=LlamaTokenizer(name_or_path='huggyllama/llama-7b', 
    # vocab_size=32000, model_max_length=512, is_fast=False, padding_side='right', truncation_side='right', 
    # special_tokens={'bos_token': AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
    #                 'eos_token': AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
    #                 'unk_token': AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True), 
    #                 'pad_token': '[PAD]'}, clean_up_tokenization_spaces=False))}
    ########## cf. 'train_dataset' goes to the following function in class Trainer, so it applies data parallelism
    # DistributedSampler(
    #     self.train_dataset,
    #     num_replicas=self.args.world_size,
    #     rank=self.args.process_index,
    #     seed=seed,
    # )
    if dist.get_rank() == 0:
        print('[2]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) # ZeRO-3: 5394 MB, ZeRO-2: 27625 MB
        print('[2]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 18.11 GB
    # trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data_module["train"],
        data_collator=data_collator,
    )
    trainer.train()
    # trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()