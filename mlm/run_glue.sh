export TASK_NAME=mrpc

python3 run_glue.py \
  --model_name_or_path /home/ubuntu/mlm/mqa_roberta_test \
  --tokenizer_name roberta-base \
  --task_name $TASK_NAME \
  --do_eval \
  --max_seq_length 512 \
  --output_dir /tmp/$TASK_NAME/