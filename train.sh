#!/bin/bash

uv run lerobot-train \
    --dataset.repo_id=kowyo/plugin-v3 \
    --policy.type=pi05_custom \
    --output_dir=./outputs/pi05_training \
    --job_name=pi05_training \
    --policy.pretrained_path=lerobot/pi05_base \
    --policy.compile_model=true \
    --policy.gradient_checkpointing=true \
    --wandb.enable=true \
    --policy.dtype=bfloat16 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --steps=3000 \
    --policy.device=cuda \
    --batch_size=32 \
    --policy.push_to_hub=true \
    --policy.repo_id=kowyo/pi05-finetuned
