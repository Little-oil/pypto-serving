# Parallel Serving Dev Notes

For standard models, serving supports a v1 `DP x TP` device topology. Data
parallelism creates one independent serving engine per replica, and tensor
parallelism passes one device group to the PyPTO L3 distributed worker for that
replica. Single-device serving remains the default.

DeepSeekV4 additionally supports model-local overlapped DP/EP placement. Its
attention DP ranks and MoE EP ranks reuse the same eight-device worker group;
see [DeepSeek V4 NPU Serving Dev Notes](model/deepseek-v4.md). Pipeline
parallelism and general-purpose expert-parallel placement are not supported yet.

Install the checkout before running the commands below:

```bash
python -m pip install --no-deps -e .
```

## DP=2 Serving

For example, run two data-parallel replicas on devices 0 and 1:

```bash
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --devices 0,1 \
  --dp 2 \
  --tp 1 \
  --max-model-len 512 \
  --max-new-tokens 16 \
  --port 8899
```

Send a completion request to the DP=2 server:

```bash
curl --noproxy "*" http://127.0.0.1:8899/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Huawei is","max_tokens":16,"temperature":0.0}'
```

Offline generation (`pypto-serving --prompt`) shares the serving engine, so
`--devices`, `--tp`, and `--dp` behave exactly as in serving: prompts are
routed across replicas with the same least-pending-tokens policy.
