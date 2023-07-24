export WANDB_MODE=offline
# --model_name_or_path huggyllama/llama-7b facebook/opt-6.7b
# prev: --learning_Rate 1e-5, --per_device_train_batch_size 8 --gradient_accumulation_steps 16
python3 -m torch.distributed.launch --nproc_per_node 8 --master_port=8888 --use-env llama_train_colo.py \
    --model_name_or_path huggyllama/llama-7b \
    --data_path togethercomputer/RedPajama-Data-1T-Sample \
    --output_dir ./trained/temp2.pt \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03
    # \
    # | tee ./logs/colo_opt-6.7b.log