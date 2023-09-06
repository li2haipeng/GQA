from transformers import AutoConfig, AutoModelForCausalLM, LlamaForCausalLM
model_config = AutoConfig.from_pretrained("NousResearch/Llama-2-7b-hf")
model_config.update({"num_key_value_heads": 8})
# print(model_config)
model = AutoModelForCausalLM.from_config(config=model_config)
# model.gradient_checkpointing_enable()   ###
# print(model)
pytorch_total_params = sum(p.numel() for n, p in model.named_parameters())
print("Param: {:.2f}".format(pytorch_total_params/1000/1000))