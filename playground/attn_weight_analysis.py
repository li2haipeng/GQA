# import wandb
# wandb.login()
# wandb.init(project="pytorch-demo", name="flash_1_eval")
from transformers import pipeline, AutoTokenizer, AutoConfig, GenerationConfig
import torch
import time
import sys
import os
sys.path.append("..")
import torch.distributed as dist
import utils
from modeling_llama2_kv_pruning import LlamaForCausalLM
# from flash_attn_llama.modeling_bifurcated_llama import LlamaForCausalLM
from transformers import LlamaTokenizer
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

kv_h=32
model_config = AutoConfig.from_pretrained("NousResearch/Llama-2-7b-hf")
model_config.update({"num_key_value_heads": kv_h})
model_config.update({"output_attentions": True})
# model

# model = LlamaForCausalLM(config=model_config)
model = LlamaForCausalLM.from_pretrained("/home/ubuntu/GQA/trained/alpaca_llama2_32/checkpoint-200", config=model_config)
model.half()
# state_dict = {} 
# from safetensors.torch import load_file  
# state_dict.update(load_file("/home/ubuntu/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc/model-00001-of-00002.safetensors"))
# state_dict.update(load_file("/home/ubuntu/.cache/huggingface/hub/models--NousResearch--Llama-2-7b-hf/snapshots/dacdfcde31297e34b19ee0e7532f29586d2c17bc/model-00002-of-00002.safetensors"))

# for n, p in model.named_parameters():
#     x = state_dict[n]
#     if not 'norm' in n and not 'bias' in n:
#         q_per_group = model_config.num_attention_heads // model_config.num_key_value_heads
#         if kv_h != 32:
#             if 'k_proj' in n or 'v_proj' in n:
#                 x = utils.group_weight(x, model_config.num_attention_heads, q_per_group)
#     p.data.copy_(x)

model.cuda()

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=1,
)

pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Param: {:.2f}".format(pytorch_total_params/1000/1000))

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")



if tokenizer.eos_token is None:
    # special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    tokenizer.add_special_tokens({'eos_token': DEFAULT_EOS_TOKEN})
if tokenizer.bos_token is None:
    # special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    tokenizer.add_special_tokens({'bos_token': DEFAULT_BOS_TOKEN})
if tokenizer.unk_token is None:
    # special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN
    tokenizer.add_special_tokens({'unk_token': DEFAULT_UNK_TOKEN})
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
model.resize_token_embeddings(len(tokenizer))

instructions = [
    # "Tell me about alpacas.",
    # "Tell me about the king of France in 2019.",
    "Tell me about the president of Mexico in 2019.",
    # "Tell me about the president of America in 2001.",
    # "What is the background of the president of France in 2001.",
    # "Who is the president of Korea in 1988.",
    # "List all Canadian provinces in alphabetical order.",
    # "Write a Python program that prints the first 10 Fibonacci numbers.",
    # "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    # "Tell me five words that rhyme with 'shock'.",
    # "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    # "Count up from 1 to 500."
    ] * 1

# instructions = ["".join(instructions)] * 1
# print(instructions)

inputs = tokenizer(instructions, return_tensors="pt")["input_ids"]


interations = 1
print("start...")
for i in range(interations):
    t0 = time.time()
    print("run:", i)

    outputs = model.generate(input_ids=inputs.cuda(),
                                    generation_config=generation_config,
                                    max_new_tokens=512,
                                    pad_token_id=tokenizer.eos_token_id)

    generated_tokens = outputs
    t1 = time.time()
    print(torch.cuda.memory_summary())
    # print("1", {torch.cuda.memory_allocated()}, {torch.cuda.memory_cached()})
    # torch.cuda.empty_cache()
    # print("2", {torch.cuda.memory_allocated()}, {torch.cuda.memory_cached()})
    # calculate metrics

    tokens_gen_text = len(generated_tokens[0])
    # print(generated_tokens.shape)
    # for i in range(generated_tokens.shape[0]):
    print("Response: {}".format(tokenizer.decode(generated_tokens[0, :])))

    throughput = (tokens_gen_text) / ((t1 - t0))

    # view results
    print(f"""Tokens generated: {tokens_gen_text}
    Time: {t1 - t0:.1f} seconds
    Tokens per second: {throughput:.1f}
    Latency: {1000 / throughput:.1f} ms""")