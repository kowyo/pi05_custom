#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
rm -rf outputs/pi05_training
rm -f train_*.log

LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"

source .venv/bin/activate

uv run accelerate launch \
    --multi_gpu \
    --num_processes=4 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    $(which lerobot-train) \
    --dataset.repo_id=kowyo/plugin-v2 \
    --dataset.revision=main \
    --policy.type=pi05_custom \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=true \
    --steps=20000 \
    --log_freq=20 \
    --policy.device=cuda \
    --batch_size=24 \
    --policy.push_to_hub=true \
    --policy.repo_id=kowyo/pi05_custom_v2 \
    2>&1 | tee "$LOG_FILE"
