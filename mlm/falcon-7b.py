from transformers import AutoModelForMaskedLM, RobertaForMaskedLM, AutoModelForCausalLM
import math
from transformers import TrainingArguments, Trainer, TrainerCallback
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer, AutoConfig, BertConfig, BertTokenizer, RobertaTokenizer, BertTokenizerFast
from datasets import load_dataset
from tokenizers import trainers, Tokenizer, normalizers, ByteLevelBPETokenizer
from pathlib import Path
import time
import torch
import sys
import os

# from torch.utils.tensorboard import SummaryWriter
language = "wikipedia_en"
model_name = "tiiuae/falcon-7b"
model = AutoModelForCausalLM.from_pretrained(model_name)
exit()
model_dir = model_name + f"-pretrained-{language}"
Path(model_dir).mkdir(parents=True, exist_ok=True)
config = AutoConfig.from_pretrained(model_name)

# writer = SummaryWriter('runs/fashion_mnist_experiment_1')

class TimeHistory(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.epoch_time_start = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_time_start = time.time()
    def on_epoch_end(self, args, state, control, **kwargs):
        print("epoch time: {}".format(time.time()-self.epoch_time_start))


def train_tokenizer(tokenizer, raw_dataset):

    def batch_iterator(batch_size=1000):
        for i in range(0, len(raw_dataset), batch_size):
            yield raw_dataset["train"][i: i + batch_size]["text"]

    tokenizer.train_from_iterator(batch_iterator(), vocab_size=config.vocab_size, min_frequency=2, 
                                  special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

    tokenizer.save_pretrained(f"{model_dir}/tokenizer.json")
    return tokenizer


def dataset_mapping(trained_tokenizer, raw_dataset, max_seq_length):
    def tokenize_function(examples):
        return trained_tokenizer(examples["text"], return_special_tokens_mask=True, truncation=True, max_length=max_seq_length)
    
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // max_seq_length) * max_seq_length
        result = {
            k: [t[i: i + max_seq_length]
                for i in range(0, total_length, max_seq_length)]
            for k, t in concatenated_examples.items()
        }
        return result
    

    tokenized_datasets = raw_dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=raw_dataset["train"].column_names)
    tokenized_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4)
    return tokenized_datasets


def mha_to_mqa(mha_model, config):

    def group_weight(w, hidden_size, q_per_group):
        w_l = list()
        g = int(hidden_size/q_per_group)
        for i in range(g):
            grouped_w = torch.mean(w[i * q_per_group: (i+1) * q_per_group, :], dim=0, keepdim=True)
            w_l.append(grouped_w)
        w_l = torch.cat(w_l, dim=0)
        return w_l

    q_per_group = int(config.num_attention_heads / config.kv_h)
    d_m = config.hidden_size

    model = AutoModelForMaskedLM.from_config(config)
    layers = mha_model.roberta.encoder.layer
    for i, l in enumerate(layers):
        w_k = l.attention.self.key.weight
        w_v = l.attention.self.value.weight
        grouped_w_k = group_weight(w_k, d_m, q_per_group)
        grouped_w_v = group_weight(w_v, d_m, q_per_group)
        model.roberta.encoder.layer[i].attention.self.key.weight = torch.nn.Parameter(grouped_w_k)
        model.roberta.encoder.layer[i].attention.self.value.weight = torch.nn.Parameter(grouped_w_v)
          
    return model


def main():
    args = sys.argv
    max_seq_length = int(args[1])
    per_device_batch_size = int(args[2])

    num_epochs = 5
    training_seed = 0
    learning_rate = 5e-5

    raw_dataset = load_dataset("wikipedia", "20220301.en")
    raw_dataset["train"] = load_dataset("wikipedia", "20220301.en", split="train[1%:]")
    raw_dataset["validation"] = load_dataset("wikipedia", "20220301.en", split="train[:1%]")
    #raw_dataset["train"] = raw_dataset["train"].select(range(10000))
    # raw_dataset["validation"] = raw_dataset["validation"].select(range(10000))

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    # tokenizer = train_tokenizer(tokenizer, raw_dataset)
    tokenized_datasets = dataset_mapping(tokenizer, raw_dataset, max_seq_length)

    # tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    pretrained_model = RobertaForMaskedLM.from_pretrained("/home/ubuntu/mlm/roberta_hf")

    config.update({"kv_h": 3})
    config.save_pretrained(f"{model_dir}")

    model = mha_to_mqa(pretrained_model, config)
    
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print("Param: {:.2f}".format(pytorch_total_params/1000/1000))

    training_args = TrainingArguments(
        output_dir="test_roberta_group_{}_checkpoint".format(config.kv_h),
        evaluation_strategy="steps",
        weight_decay=0.01,
        # bf16=True,
        fp16=True,
        do_eval=False,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        seed=training_seed,
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
    # trainer.train()

    # eval_results = trainer.evaluate()
    # print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}, samples_per_sec: {eval_results['eval_samples_per_second']}")
    
    trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}, samples_per_sec: {eval_results['eval_samples_per_second']}")
    trainer.save_model("roberta_group_{}".format(config.kv_h))
    
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # print("Param: {:.2f}".format(pytorch_total_params/1000/1000))


if __name__ == "__main__":
    main()
