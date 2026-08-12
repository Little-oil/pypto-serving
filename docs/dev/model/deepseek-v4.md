# DeepSeek V4 NPU Serving Dev Notes

These commands are for DeepSeek V4 Flash W8A8 serving checks on shared Ascend
development machines with `task-submit`. Run them from the pypto-serving
checkout.

## 8-Device Offline Generation

The offline entry uses the same scheduler, worker process, rank-partitioned
cache pools, and MTP acceptance path as HTTP serving, without opening a port:

```bash
task-submit --device 8,9,10,11,12,13,14,15 --max-time 0 --timeout 0 --ptoas 0.48 --run "PYPTO_RUNTIME_LOG=error PTO2_RING_DEP_POOL=131072 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_HEAP=2147483648 PTO2_OP_EXECUTE_TIMEOUT_US=400000000 PTO2_STREAM_SYNC_TIMEOUT_MS=440000 PTO2_SCHEDULER_TIMEOUT_MS=320000 SERVING_WORKER_STEP_TIMEOUT=1800 python examples/model/deepseek_v4/npu_generate.py --model-dir /data/models/dsv4-flash-w8a8 --prompt 'Huawei is' --platform a2a3 --devices 8,9,10,11,12,13,14,15 --max-seq-len 512 --max-new-tokens 20 --enable-mtp"
```

Use `--num-prompts N` to exercise continuous batching, or add
`--profile --profile-output /path/to/profile` to capture only the generation
window after model initialization.

## 8-Device DP/EP Serving

Use the quantized checkpoint under `/data/models/dsv4-flash-w8a8` and run with
overlapped attention DP=8 and MoE EP=8 on devices 8-15. Both parallel axes use
the same eight physical ranks, so this is one model replica rather than eight
independent serving replicas:

```bash
task-submit --device 8,9,10,11,12,13,14,15 --max-time 0 --timeout 0 --ptoas 0.48 --run "PYPTO_RUNTIME_LOG=error PTO2_RING_DEP_POOL=131072 PTO2_RING_TASK_WINDOW=131072 PTO2_RING_HEAP=2147483648 PTO2_OP_EXECUTE_TIMEOUT_US=400000000 PTO2_STREAM_SYNC_TIMEOUT_MS=440000 PTO2_SCHEDULER_TIMEOUT_MS=320000 SERVING_WORKER_STEP_TIMEOUT=1800 pypto-serving --model /data/models/dsv4-flash-w8a8 --served-model-name dsv4-flash-w8a8 --backend npu --platform a2a3 --devices 8,9,10,11,12,13,14,15 --dp 8 --ep 8 --tp 1 --block-size 128 --max-model-len 512 --max-num-seqs 32 --max-num-batched-tokens 512 --long-prefill-token-threshold 2048 --enable-mtp --no-enable-prefix-caching --port 8225 --show-startup-logs"
```

Each NPU runs one prefill row at a time, so DP=8 admits up to eight prefill
requests in one global step. MTP decode uses B4S2 on each rank, for a maximum of
32 global active rows. The scheduler may admit fewer long-context requests when
a rank's fixed pypto-lib cache pools are full.

For repeated launches, set `PYPTO_PROG_BUILD_DIR` to a persistent directory and
add `--use-compile-cache`. The first launch populates a device-specific worker
subdirectory after executable assembly. Later launches reuse the compiled
programs without fingerprint validation, so use the same model configuration,
assigned devices, and kernel sources, and clear the directory after any change.

MTP prefill context, draft token, committed tail, and acceptance counters are
owned by request ID. MTP prefill and decode share one worker-resident cache, but
each request addresses it with the scheduler-owned rank-local `ori` block IDs.
The scheduler reserves the extra speculative position before dispatch, including
when a draft crosses a 128-token page boundary.

Before the first decode is prepared, each request owns a stable rank-local
device-state slot and reuse generation. Terminal prefill fills that reserved
slot with the committed tail token, next draft token, tail position, and
committed count. The fused decode kernel
uses `(rank, slot, generation)` to build the next `[tail, draft]` input rows and
sequence metadata before main decode, then updates the same slot after MTP
verification. Host output processing mirrors the state for scheduling and
statistics, but is not an input dependency of the next steady-state decode.
Generation matching prevents a stale queued step from updating a slot after
preemption and reuse.

The seven main-model KV/state pools are allocated during runner preflight as
rank-sharded worker-resident tensors. Prefill and decode pass the same device
handles and address them with scheduler-owned group block IDs; there is no
prefill CPU snapshot or cache handoff. Reassigned pages are cleared with
targeted host-to-device copies before their new owner writes them.

## Optional Prepacked Weights

The 43 hidden layers can be converted once into the final rank-stacked Host
layout:

```bash
pypto-prepack-deepseek-v4 /data/models/dsv4-flash-w8a8
```

The command atomically writes
`pypto-deepseek-v4-stacked-r8.safetensors` beside the checkpoint. Subsequent
starts sample its Linux page-cache residency before opening it. A hot sidecar is
validated against the checkpoint-file and deployment fingerprint, then
memory-mapped as the final layout instead of repacking every hidden layer. A
cold, missing, or stale sidecar uses the original checkpoint path, avoiding a
cold 323 GiB page-fault stream on the weight-upload path. Rebuild with `--force`
after replacing checkpoint shards or changing the packed rank layout.

## Completion Check

Check server health first:

```bash
curl --noproxy "*" http://127.0.0.1:8225/health
```

Then send a deterministic completion request:

```bash
curl --noproxy "*" -s http://127.0.0.1:8225/v1/completions -H "Content-Type: application/json" -d '{"model":"dsv4-flash-w8a8","prompt":"Huawei is","max_tokens":25,"temperature":0.0}'
```
