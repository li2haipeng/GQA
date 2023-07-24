from dataclasses import dataclass, field
import GPUtil
import numpy as np
import torch
import transformers
from transformers import GenerationConfig
import colossalai
from colossalai.zero import ColoInitContext
from colossalai.utils import get_current_device, print_rank_0
from colossalai.tensor import ProcessGroup, ShardSpec, ComputePattern, ComputeSpec
from colossalai.nn.parallel.layers import init_colo_module
from colossalai.cluster import DistCoordinator
import torch.distributed as dist
from colossalai.logging import disable_existing_loggers, get_dist_logger
import sys
sys.path.append('..')

from transformers import AutoConfig
# LLaMA
import transformers.models.llama.modeling_llama
from transformers.models.llama.modeling_llama import LlamaForCausalLM
# OPT
import transformers.models.opt.modeling_opt
from transformers.models.opt.modeling_opt import OPTForCausalLM

from llama_train_colo import ModelArguments, smart_tokenizer_and_embedding_resize, DEFAULT_PAD_TOKEN, DEFAULT_EOS_TOKEN, \
    DEFAULT_BOS_TOKEN, DEFAULT_UNK_TOKEN, PROMPT_DICT


@dataclass
class InferenceArguments:
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={"help": "Load the model in 8-bit mode."},
    )
    inference_dtype: torch.dtype = field(
        default=torch.float16,
        metadata={"help": "The dtype to use for inference."},
    )
    override_checkpoint: str = field(
        default=None,
        metadata={"help": "Name of the checkpoint file to override."},
    )


def generate_prompt(instruction, input=None):
    if input:
        return PROMPT_DICT["prompt_input"].format(instruction=instruction, input=input)
    else:
        return PROMPT_DICT["prompt_no_input"].format(instruction=instruction)


def inference():
    tp_degree = 8
    dp_degree = 1
    # 0: by row (bs=8, peak_mem=28487 MB), -1: by col (bs=8, peak_mem=24855 MB)
    dims = -1
    disable_existing_loggers()
    import transformers.models.llama.modeling_llama
    colossalai.launch_from_torch(config=dict(parallel=dict(data=dp_degree, pipeline=1,
                                                           tensor=dict(size=tp_degree, mode='1d'))))
    parser = transformers.HfArgumentParser((ModelArguments, InferenceArguments))
    model_args, inference_args = parser.parse_args_into_dataclasses()
    logger = get_dist_logger()
    shard_pg = ProcessGroup(tp_degree=tp_degree)
    default_dist_spec = ShardSpec([dims], [tp_degree])

    with ColoInitContext(device=get_current_device(), default_dist_spec=default_dist_spec, default_pg=shard_pg):
        model_config = AutoConfig.from_pretrained("/home/ubuntu/gqa/llm/gqa_llama")
        # if 'llama-7b' in model_args.model_name_or_path:
        #     model = LlamaForCausalLM(model_config)
        # elif 'opt-6.7b' in model_args.model_name_or_path:
        #     model = OPTForCausalLM(model_config)
        # model = transformers.AutoModelForCausalLM.from_pretrained(
        #   model_args.model_name_or_path,
        #   # load_in_8bit=inference_args.load_in_8bit,
        #   # torch_dtype=inference_args.inference_dtype,
        #   # device_map="auto",
        # )
        model_config.update({"kv_h": 8})
        model = transformers.AutoModelForCausalLM.from_config(model_config)

        generation_config = GenerationConfig(
            temperature=0.1,
            top_p=0.75,
            num_beams=4,
        )

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            use_fast=False,
            model_max_length=inference_args.model_max_length,
        )

        if tokenizer.pad_token is None:
            # For size matching of Colossal-AI
            tokenizer.pad_token = tokenizer.eos_token
            # # Other cases
            # smart_tokenizer_and_embedding_resize(
            #   special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            #   tokenizer=tokenizer,
            #   model=model,
            # )
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    compute_spec = ComputeSpec(ComputePattern.TP1D)
    init_colo_module(model, compute_spec, pg=shard_pg, recursive=True)
    if inference_args.override_checkpoint is not None:
        logger.info("Loading override checkpoint.", ranks=[0])
        try:
            state_dict = torch.load(inference_args.override_checkpoint + str(dist.get_rank()) + '.pt')
            model.load_state_dict(state_dict)
            pytorch_total_params = sum(p.numel() for p in model.parameters())
            print("Param: {:.2f}".format(tp_degree * pytorch_total_params/1000/1000))
        except:
            raise Exception("Failed to load checkpoint")
        model.cuda()
        model.half()
        model.eval()

        # ctx = ""
        import time
        start = time.time()
        for instruction in [
            "My name is Anna, and I ",
            "Tell me about the president of Mexico in 2019.",
            # "Tell me about the king of France in 2019.",
            # "List all Canadian provinces in alphabetical order.",
            # "Write a Python program that prints the first 10 Fibonacci numbers.",
            # "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",
            # "Tell me five words that rhyme with 'shock'.",
            "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
            # "Count up from 1 to 500.",
        ]:
            inputs = tokenizer(generate_prompt(
                instruction, None), return_tensors="pt")
            dist.broadcast(inputs['input_ids'].cuda(), src=0)
            if dist.get_rank() == 0:
                # logger.info("Instruction: {}".format(instruction), ranks=[0])
                print("Instruction: {}".format(instruction))
                outputs = model.generate(input_ids=inputs["input_ids"].cuda(),
                                         generation_config=generation_config,
                                         max_new_tokens=inference_args.model_max_length,
                                         return_dict_in_generate=True,
                                         output_scores=True)
                input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                generated_tokens = outputs.sequences[:, input_length:]
                print(print_rank_0('Used GPU mem: {0}'.format(GPUtil.getGPUs()[0].memoryUsed)))
                # ctx += f"Instruction: {instruction}\n" + f"Response: {generated_tokens[0]}\n"
                # logger.info("Response: {}".format(tokenizer.decode(generated_tokens[0])), ranks=[0])
                print("Response: {}".format(
                    tokenizer.decode(generated_tokens[0])))
                print()
            else:
                model.generate(input_ids=inputs["input_ids"].cuda(),
                               generation_config=generation_config,
                               max_new_tokens=inference_args.model_max_length,
                               return_dict_in_generate=True,
                               output_scores=True)
            print("time: {:.2f}".format(time.time()-start))
            start = time.time()

if __name__ == "__main__":
    inference()
