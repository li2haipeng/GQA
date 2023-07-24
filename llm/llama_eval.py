from collections import OrderedDict
import os
from transformers import get_cosine_schedule_with_warmup
from transformers import Trainer, DataCollatorForLanguageModeling, AutoConfig, TrainingArguments
from datasets import load_dataset
from torch.utils.data import Dataset
import math
import utils
import transformers
import torch
import time
import sys
sys.path.append('..')
IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

state_path = "/home/ubuntu/gqa/trained/shard_colo_llama-7b"
state_lists = os.listdir(state_path)
state_lists = sorted(state_lists, key=lambda x: int(os.path.splitext(x)[0]))
trained_state = OrderedDict()
for f in state_lists:
    state = os.path.join(state_path, f)
    # print(state)
    sharded_state = torch.load(state)
    # print(sharded_state.get_device())
    for k, v in sharded_state.items():
        if k in trained_state.keys() and len(v.shape) > 1:
            # print(k, v.shape)
            vv = trained_state[k]
            # print(torch.equal(vv,v))
            trained_state[k] = torch.cat((vv, v), dim=-1)
        else:
            trained_state[k] = v
# for k,v in trained_state.items():
#     print(k, v.shape)
    # state_dict = torch.load(stat_path)
# print(state_dict.keys())
# sys.exit()


model_config = AutoConfig.from_pretrained("gqa_llama")
model = transformers.AutoModelForCausalLM.from_config(model_config)
model.load_state_dict(trained_state)
model.half()
# make sure token embedding weights are still tied if needed
# model.tie_weights()

# Set model in evaluation mode to deactivate DropOut modules by default
# model.eval()
tokenizer = transformers.AutoTokenizer.from_pretrained(
    "huggyllama/llama-7b",
    model_max_length=512,
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

data_module = load_dataset("togethercomputer/RedPajama-Data-1T-Sample")
# data_module["train"] = load_dataset("togethercomputer/RedPajama-Data-1T-Sample", split="train[:1%]")
data_module["validation"] = load_dataset(
    "togethercomputer/RedPajama-Data-1T-Sample", split="train[1%:2%]")
data_module["validation"] = data_module["validation"].select(range(1000))
tokenized_datasets = utils.dataset_mapping(
    tokenizer, data_module, max_seq_length=2048)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
training_args = TrainingArguments(
    output_dir="test_group_{}_checkpoint".format(model_config.kv_h),
    evaluation_strategy="steps",
    weight_decay=0.01,
    # bf16=True,
    fp16=True,
    do_eval=False,
    per_device_train_batch_size=16,
    max_steps=500,
    push_to_hub=False,
    adam_beta2=0.98,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
)

# trainer.save_model("roberta_hf")

eval_results = trainer.evaluate()
print(
    f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}, samples_per_sec: {eval_results['eval_samples_per_second']}")
# trainer.save_model("roberta_group_{}".format(config.kv_h))

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Param: {:.2f}".format(pytorch_total_params/1000/1000))
