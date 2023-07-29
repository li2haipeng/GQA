# python3 -m torch.distributed.launch --nproc_per_node=8 --nnodes=2 --node_rank=0 \
#      --master_addr="172.31.39.37" --master_port=8887 --use-env colo_inference.py \
#     --model_name_or_path huggyllama/llama-7b \
#     --override_checkpoint /home/ubuntu/GQA/trained/gqa_1_shard_colo_llama-7b/


python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=8887 --use-env pipeline_inf.py \
    --model_name_or_path huggyllama/llama-7b \
    --override_checkpoint /home/ubuntu/GQA/trained/test_gqa_32_shard_colo_llama-7b/