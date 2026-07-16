# Qwen NPU Serving Dev Notes

These commands are for Qwen3 serving checks on the shared Ascend development
machines that provide `task-submit`. Use the README commands for environment-neutral usage.

## Single-Device Serving

```bash
task-submit --device auto --max-time 1200 --run \
  "pypto-serving \
    --model /data/models/Qwen3-14B \
    --backend npu \
    --platform a2a3 \
    --devices {} \
    --max-model-len 512 \
    --port 8899"
```

## DP=2 Serving

```bash
task-submit --device auto --device-num 2 --max-time 1800 --run \
  "pypto-serving \
    --model /data/models/Qwen3-14B \
    --backend npu \
    --platform a2a3 \
    --devices {} \
    --dp 2 \
    --tp 1 \
    --max-model-len 512 \
    --port 8899"
```
