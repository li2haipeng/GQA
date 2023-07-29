import dataclasses
import logging
import math
import os
import io
import sys
import time
import json
from typing import Optional, Sequence, Union
import torch
import transformers as tf
from collections import OrderedDict

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def dataset_mapping(trained_tokenizer, raw_dataset, max_seq_length):
    def tokenize_function(examples):
        return trained_tokenizer(examples["ctx"], return_special_tokens_mask=True, truncation=True, max_length=max_seq_length)
    
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


# def group_weight(w, hidden_s, q_per_group):
#         w_l = list()
#         g = int(hidden_s / q_per_group)
#         if len(w.shape) == 2:
#             for i in range(g):
#                 a = w[i * q_per_group: (i + 1) * q_per_group, :]
#                 grouped_w = torch.mean(a, dim=0, keepdim=True)
#                 w_l.append(grouped_w)
#             w_l = torch.cat(w_l, dim=0)
#         elif len(w.shape) == 1:
#             for i in range(g):
#                 a = w[i * q_per_group: (i + 1) * q_per_group]
#                 grouped_w = torch.mean(a)
#                 w_l.append(grouped_w)
#             w_l = torch.stack(w_l)
#         # equal = torch.equal(w, w_l)

#         return w_l

def group_weight(w, q_h, q_per_group):
        w_l = list()
        g = int(q_h / q_per_group)
        assert len(w.shape) == 2
        
        w = w.chunk(g, dim = 0)
        for i in range(g):
            qw = list(w[i].chunk(q_per_group, dim = 0))
            grouped_w = torch.sum(torch.stack(qw), dim=0) / q_per_group
            w_l.append(grouped_w)
        w_l = torch.cat(w_l, dim=0)

        return w_l


def mha_to_mqa(mha_model, config):

    kv_h = config.kv_h
    hidden_size = config.hidden_size
    head_size = int(hidden_size / kv_h)
    q_per_group = int(config.num_attention_heads / kv_h)

    model = tf.AutoModelForMaskedLM.from_config(config)
    sd = model.state_dict()
    pretrained_state = mha_model.state_dict().copy()
    assert len(pretrained_state) == len(sd)
    new_state = OrderedDict()
    for pre_s, s in zip(pretrained_state.items(), sd.items()):
        if pre_s[1].shape == s[1].shape:
            new_state[s[0]] = pre_s[1]
        else:
            grouped_s = group_weight(pre_s[1], hidden_size, q_per_group)
            new_state[s[0]] = grouped_s
    model.load_state_dict(new_state)
    return model


def gqa_weight(pretrained_state, gqa_state, config, tp_degree):
    kv_h = config.kv_h
    hidden_size = config.hidden_size
    head_size = int(hidden_size / kv_h)
    q_per_group = int(config.num_attention_heads / kv_h)
    assert len(pretrained_state) == len(gqa_state)
    new_state = OrderedDict()
    for pre_s, s in zip(pretrained_state.items(), gqa_state.items()):
        if pre_s[1].shape == s[1].shape:
            new_state[s[0]] = pre_s[1]
        else:
            grouped_s = group_weight(pre_s[1], hidden_size, q_per_group)
            new_state[s[0]] = grouped_s
    return new_state