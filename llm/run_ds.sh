export WANDB_MODE=online
# --model_name_or_path facebook/opt-6.7b \
# python3 -m torch.distributed.launch --nproc_per_node=8 --nnode 2 --node_rank=0 \
#     --master_addr="172.31.39.37" --master_port=8888 --use-env llama_train_ds.py \
#     --model_name_or_path huggyllama/llama-7b \
#     --kv_h 8 \
#     --bf16 True \
#     --output_dir /home/ubuntu/GQA/trained/DS_llama \
#     --num_train_epochs 20 \
#     --per_device_train_batch_size 32 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 200 \
#     --save_total_limit 1 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 5 \
#     --deepspeed ds_config.json


python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=8888 --use-env llama_train_ds.py \
    --model_name_or_path huggyllama/llama-7b \
    --kv_h 8 \
    --bf16 True \
    --output_dir /home/ubuntu/GQA/trained/DS_llama \
    --num_train_epochs 20 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 5 \
    --deepspeed ds_config.json
