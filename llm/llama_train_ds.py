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
sys.path.append("..")
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
import utils
from torch.utils.data import Dataset
from transformers import Trainer, AutoConfig, DataCollatorForLanguageModeling
from flash_attn_llama.modeling_flash_llama import LlamaForCausalLM
from datasets import load_dataset
import torch.distributed as dist
import GPUtil
import psutil
import math

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
    kv_h: int = field(default=32)


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
    evaluation_strategy="steps",
    # weight_decay=0.01,
    # bf16=True,
    fp16=True,
    # per_device_train_batch_size=2,
    # learning_rate=learning_rate,
    max_steps=2000,
    seed=0,
    push_to_hub=False,
    # adam_beta2=0.98,
    remove_unused_columns=False,
    report_to="wandb",





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
    kv_h = model_args.kv_h
  
    if dist.get_rank() == 0:
        print('[0]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed))    # 3 MB
        print('[0]Virtual total mem: {0}'.format(get_size(psutil.virtual_memory().total))) # 1.10 TB
        print('[0]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 6.20 GB
    model_config = AutoConfig.from_pretrained(model_args.model_name_or_path)

    model_config.update({"kv_h": kv_h})

    model = LlamaForCausalLM(model_config)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Param: {:.2f}".format(pytorch_total_params/1000/1000))
    model.gradient_checkpointing_enable()   ###
    state_dict = {}
      
    from safetensors.torch import load_file  
    state_dict.update(load_file("/home/ubuntu/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/model-00001-of-00002.safetensors"))
    state_dict.update(load_file("/home/ubuntu/.cache/huggingface/hub/models--huggyllama--llama-7b/snapshots/8416d3fefb0cb3ff5775a7b13c1692d10ff1aa16/model-00002-of-00002.safetensors"))
    
    for n, p in model.named_parameters():
        x = state_dict[n]
        if not 'norm' in n and not 'bias' in n:
            q_per_group = model_config.num_attention_heads // model_config.kv_h
            if kv_h != 32:
                if 'k_proj' in n or 'v_proj' in n:
                    x = utils.group_weight(x, model_config.num_attention_heads, q_per_group)
        p.data.copy_(x)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        # special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
        tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    if tokenizer.eos_token is None:
        # special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
        tokenizer.add_special_tokens({'eos_token': DEFAULT_EOS_TOKEN})
    if tokenizer.bos_token is None:
        # special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
        tokenizer.add_special_tokens({'bos_token': DEFAULT_BOS_TOKEN})
    if tokenizer.unk_token is None:
        # special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
        tokenizer.add_special_tokens({'unk_token': DEFAULT_UNK_TOKEN})


    if dist.get_rank() == 0:
        print('[1]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed))    # ZeRO-3: 5449 MB, ZeRO-2: 27625 MB
        print('[1]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 5.65 GB

    raw_dataset = load_dataset("wikipedia", "20220301.en")
    raw_dataset["train"] = load_dataset("wikipedia", "20220301.en", split="train[1%:10%]").select(range(10000))
    #raw_dataset["train"] = raw_dataset["train"].select(range(10000))
    # raw_dataset["validation"] = load_dataset("wikipedia", "20220301.en", split="train[:1%]")
    data_module = utils.dataset_mapping(tokenizer, raw_dataset, max_seq_length=2048)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if dist.get_rank() == 0:
        print('[2]Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)) # ZeRO-3: 5394 MB, ZeRO-2: 27625 MB
        print('[2]Virtual used mem: {0}'.format(get_size(psutil.virtual_memory().used))) # 18.11 GB
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, 
                      train_dataset=data_module["train"], 
                    #   eval_dataset=data_module["validation"], 
                      data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}, samples_per_sec: {eval_results['eval_samples_per_second']}")


if __name__ == "__main__":
    train()