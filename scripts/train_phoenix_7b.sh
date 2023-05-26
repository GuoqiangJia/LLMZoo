model_name_or_path=bigscience/bloomz-7b1-mt
model_max_length=2048
data_path=/content/drive/MyDrive/bloom-training/train/data/data.json
output_dir=checkpoints/phoenix_7b/

#accelerate launch --config_file ./scripts/accelerate_config.yaml train.py \
python train.py \
  --model_name_or_path ${model_name_or_path} \
  --model_max_length ${model_max_length} \
  --data_path ${data_path} \
  --output_dir ${output_dir} \
  --bf16 True \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 2 \
  --gradient_accumulation_steps 4 \
  --save_strategy "steps" \
  --save_steps 500 \
  --evaluation_strategy "no" \
  --save_total_limit 3 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --tf32 True \
  --gradient_checkpointing True \
  --lora True \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05
