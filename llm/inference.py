import wandb
wandb.login()
wandb.init(project="pytorch-demo", name="flash_32_eval")
from transformers import pipeline, AutoTokenizer, AutoConfig, GenerationConfig, AutoModelForCausalLM
import torch
import time
import deepspeed
from deepspeed.utils.zero_to_fp32 import load_state_dict_from_zero_checkpoint
import sys
sys.path.append("..")
from flash_attn_llama.modeling_flash_llama import LlamaForCausalLM
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

model_config = AutoConfig.from_pretrained("huggyllama/llama-7b")
model_config.update({"kv_h": 32})

# model = LlamaForCausalLM(model_config)
model = LlamaForCausalLM.from_pretrained("/home/ubuntu/GQA/trained/DS_llama_32/checkpoint-100")
# model = LlamaForCausalLM.from_pretrained("huggyllama/llama-7b", config=model_config)
generation_config = GenerationConfig(
    temperature=0.1,
    top_p=0.75,
    num_beams=4,
)

ds_engine = deepspeed.init_inference(model,
                                 mp_size=1,
                                 dtype=torch.half,
                                 replace_with_kernel_inject=True)
model = ds_engine.module
# create pipeline
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
# pipe = pipeline("text-generation",
#                 model=model,
#                 tokenizer=tokenizer,
#                 device_map="auto",
#                 torch_dtype=torch.float16)

# prompt

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
    "Tell me about alpacas.",
    # "Tell me about the king of France in 2019.",
    "Tell me about the president of Mexico in 2019.",
    # "List all Canadian provinces in alphabetical order.",
    # "Write a Python program that prints the first 10 Fibonacci numbers.",
    # "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
    # "Tell me five words that rhyme with 'shock'.",
    # "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    "Count up from 1 to 500."
    ]
# inputs = []
# for i in instructions:
#     inp = tokenizer(i, return_tensors="pt")["input_ids"]
#     inputs.append(inp)
#     # dist.broadcast(inp.cuda(), src=0)
# max_len = max([i.shape[1] for i in inputs])
# inputs = [torch.nn.functional.pad(x, pad=(0, max_len - x.numel()), mode='constant', value=0) for x in inputs]
# inputs = torch.cat(inputs, dim=0)
inputs = tokenizer(instructions, return_tensors="pt", padding=True)["input_ids"]
# print(inputs.shape, max_len)
# sys.exit()
# generate text (500 new tokens)
t0 = time.time()
# result = pipe(start_text, max_new_tokens=new_tokens)
# inputs = tokenizer(start_text, return_tensors="pt")
outputs = model.generate(input_ids=inputs.cuda(),
                            generation_config=generation_config,
                            max_new_tokens=1024,
                            return_dict_in_generate=True,
                            output_scores=True)
# input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
# generated_tokens = outputs.sequences[:, input_length:]
generated_tokens = outputs.sequences

t1 = time.time()

# calculate metrics
# gen_text = generated_tokens
# tokens_gen_text = len(tokenizer(gen_text, return_tensors="pt").input_ids[0])
tokens_gen_text = len(generated_tokens[0])
print(generated_tokens.shape)
for i in range(generated_tokens.shape[0]):
    print("Response: {}".format(tokenizer.decode(generated_tokens[i, :])))

throughput = (tokens_gen_text) / (t1 - t0)

# view results
print(f"""Tokens generated: {tokens_gen_text}
Time: {t1 - t0:.1f} seconds
Tokens per second: {throughput:.1f}
Latency: {1000 / throughput:.1f} ms""")