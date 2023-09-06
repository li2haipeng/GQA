export WANDB_MODE=offline
# --model_name_or_path facebook/opt-6.7b \
torchrun --nproc_per_node=8 taeho_llama_train_ds.py \
    --model_name_or_path NousResearch/Llama-2-7b-hf \
    --kv_h 8 \
    --data_path ../alpaca_data.json \
    --bf16 True \
    --output_dir /home/ubuntu/GQA/trained/alpaca_llama2_8_flash \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --deepspeed ds_config.json

# torchrun --nproc_per_node=1 llama_train_ds.py \
#     --model_name_or_path NousResearch/Llama-2-7b-hf \
#     --kv_h 32 \
#     --bf16 True \
#     --output_dir /home/ubuntu/GQA/trained/llama2_32 \
#     --max_steps 1000 \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 4 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 50 \
#     --save_total_limit 1 \
#     --learning_rate 3e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --run_name llama2_32 \
#     --deepspeed ds_config.json \
    
