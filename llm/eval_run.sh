python3 -m torch.distributed.launch --nproc_per_node 8 --use-env colo_inference.py \
    --model_name_or_path huggyllama/llama-7b \
    --override_checkpoint /home/ubuntu/GQA/trained/gqa_8_shard_colo_llama-7b/