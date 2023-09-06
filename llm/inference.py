from transformers import pipeline, AutoTokenizer, AutoConfig, GenerationConfig, AutoModelForCausalLM
import torch
import time
import deepspeed
import sys
import os
sys.path.append("..")
import torch.distributed as dist
import utils
from flash_attn_llama.modeling_flash_llama_2 import LlamaForCausalLM
from transformers import LlamaTokenizer
from taeho_llama_train_ds import PROMPT_DICT
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

kv_h=int(sys.argv[1])
batch_size = int(sys.argv[2])
print("kv_h: {}, batch: {}".format(kv_h, batch_size))
model_config = AutoConfig.from_pretrained("NousResearch/Llama-2-7b-hf")
# model_config = AutoConfig.from_pretrained("huggyllama/llama-7b")
model_config.update({"num_key_value_heads": kv_h})
# model_config.update({"kv_h": kv_h})
print(model_config)

# model = LlamaForCausalLM(config=model_config)
model = LlamaForCausalLM.from_pretrained("/home/ubuntu/GQA/trained/alpaca_llama2_8/checkpoint-300", config=model_config)
model.cuda().half()


def generate_prompt(instruction, input=None):
  if input:
    return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
  else:
    return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)



os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=1,
)


pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Param: {:.2f}".format(pytorch_total_params/1000/1000))

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")



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
    "Tell me about the president of Mexico in 2019.",
    ] * 1
# instructions = ["".join(instructions)] * batch_size
# print(instructions)

inputs = tokenizer(generate_prompt(instructions, None), return_tensors="pt")["input_ids"]

print(inputs.size())
interations = 3
print("start...")
for i in range(interations):
    t0 = time.time()
    print("run:", i)

    outputs = model.generate(input_ids=inputs.cuda(),
                                    generation_config=generation_config,
                                    max_new_tokens=256,
                                    pad_token_id=tokenizer.eos_token_id)

    generated_tokens = outputs
    t1 = time.time()
    print(torch.cuda.memory_summary())

    tokens_gen_text = len(generated_tokens[0])

    print("Response: {}".format(tokenizer.decode(generated_tokens[0, :])))
    throughput = (tokens_gen_text) / ((t1 - t0))

    # view results
    print(f"""Tokens generated: {tokens_gen_text}
    Time: {t1 - t0:.1f} seconds
    Tokens per second: {throughput:.1f}
    Latency: {1000 / throughput:.1f} ms""")
