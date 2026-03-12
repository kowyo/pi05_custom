#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
rm -rf outputs/pi05_training

LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG_FILE"

uv run accelerate launch --multi_gpu --num_processes=8 .venv/bin/lerobot-train \
    --dataset.repo_id=kowyo/plugin-v4 \
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
    --steps=3000 \
    --batch_size=8 \
    --policy.push_to_hub=true \
    --policy.repo_id=kowyo/pi05-finetuned \
    2>&1 | tee "$LOG_FILE"
