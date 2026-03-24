# pi05_custom

A custom [pi0.5](https://www.physicalintelligence.company/blog/pi05) policy for [LeRobot](https://github.com/huggingface/lerobot).

The policy uses a PaliGemma VLM backbone with a separate action expert, and predicts actions via flow matching.

## Setup

```bash
make prepare   # install dependencies and Git hooks
```

## Training

```bash
bash train.sh
```
