---
name: qwen3-14b-online-perf-test
description: Test and analyze the ONLINE serving performance of Qwen3-14B on the pypto-serving HTTP server. The primary skill for online Qwen / serving performance testing. It starts a profiled server using the built-in SA_PROFILE Chrome-trace recorder, drives load, and produces a per-operator / per-kernel time breakdown (prefill vs decode, device-side kernel duration) so you can see where serving time goes. Use when the user wants to test online serving performance, get an operator/kernel-level timing breakdown, or troubleshoot a profile that captured no kernel events. Covers enabling profiling, starting the server, driving workload, merging trace fragments, and aggregating operator duration from the trace. For pure end-to-end throughput / latency / TTFT numbers use `vllm-bench-perf`; for offline single-generation profiling use `qwen3-14b-offline-perf-test`.
---

# Qwen3-14B online serving performance profiling

`SA_PROFILE` is the built-in Chrome trace-event recorder in `pypto_serving.tools.profile`. It is disabled by default with low overhead, and when enabled it records duration spans from the HTTP API, scheduler, engine, executor, worker, and NPU kernel-dispatch paths. Each process writes its own JSON Lines fragment; on a graceful shutdown the entry point merges them into a single `trace.json` that can be opened in a trace viewer such as Perfetto.

This skill measures **where time goes** at operator granularity during a real **online serving** run — the complement of `vllm-bench-perf` (end-to-end serving latency/throughput) and `ais-bench-eval` (accuracy). It profiles the HTTP server only; for offline single-generation profiling use the `qwen3-14b-offline-perf-test` skill instead. The canonical reference is `docs/dev/profile.md`; re-read it if the env semantics below disagree with the checkout.

Do not hard-code a specific commit, user directory, device id, port, or one-off output path. Use the model dir, output path, port, and workload the user provides, then print what actually ran.

---

## 1. Prerequisites

- A Conda environment with `pypto-serving` installed so the `pypto-serving` console script is on `PATH` (it calls `pypto_serving.cli.main:main`).
- The model weights at a local model dir (for example Qwen3-14B).
- An available NPU device. If the box gates device access through a queue wrapper such as `task-submit`, run the server inside that wrapper — `--device auto` fills the `{}` placeholder in `--devices {}` with the assigned device.
- A free TCP port. The default is `8000`; if it is in use (common when a teammate is also serving), pass `--port <other>`.

## 2. Enable profiling

Profiling turns on when **either** `SA_PROFILE_OUTPUT` or `SA_PROFILE_LEVEL` is present. For operator timing use:

| Variable | Value for this skill |
| --- | --- |
| `SA_PROFILE_OUTPUT` | An **absolute** directory path, fresh per run. A new main process clears stale `trace.*.jsonl` in its `fragments/` dir, so reusing a path overwrites the prior run. |
| `SA_PROFILE_LEVEL` | `verbose` (all levels) or `e2e,kernel`. The `kernel` level is required — without it no `kernel.*_fwd` spans are recorded and there is no operator breakdown. |

Output layout for a directory output:

```text
<SA_PROFILE_OUTPUT>/
├── fragments/trace.<pid>.jsonl   # one JSONL fragment per process (API + worker)
└── trace.json                    # merged trace, written on graceful shutdown
```

Keep whatever `PTO2_*` runtime env vars the run template uses (ring heap, op-execute/stream-sync timeouts). They are unrelated to profiling but required for the NPU runtime on this box.

## 3. Start the server

Run the server with profiling on. Template (queue-wrapped; the `--devices {}` placeholder is filled by `--device auto`):

```bash
task-submit --device auto --run --max-time 0 --timeout 0 \
"SA_PROFILE_OUTPUT=/abs/path/profile-out \
SA_PROFILE_LEVEL=verbose \
PTO2_OP_EXECUTE_TIMEOUT_US=50000000 PTO2_STREAM_SYNC_TIMEOUT_MS=55000 \
PTO2_RING_HEAP=2147483648 PTO2_RING_TASK_WINDOW=262144 PTO2_RING_DEP_POOL=262144 \
pypto-serving \
    --model /path/to/Qwen3-14B \
    --platform a2a3 \
    --port 8899 \
    --devices {} \
    --max-num-batched-tokens 4096 --max-num-seqs 16 \
    --npu-memory-utilization 0.9 --max-model-len 4096 \
    --no-enable-prefix-caching --long-prefill-token-threshold 2048"
```

The same `SA_PROFILE_*` env vars also work for the offline entry `python examples/model/qwen3_14b/npu_generate.py`, but this skill is scoped to the **online HTTP server**; do not use the offline entry here.

Wait for `INFO: Application startup complete.` / `Uvicorn running on http://0.0.0.0:<port>` before sending traffic. The worker prints `Worker entering busy loop` and the engine prints `Engine loop started` once the model and KV cache are ready.

## 4. Confirm ready, then drive workload

**Workload spec — two configs.** Each config is `input/output/num-prompts` (token lengths and request count). With all requests firing at once the load reaches ~16 concurrent and fills the `--max-num-seqs 16` batch, so `kernel.decode_fwd` spans reflect batched decode.

| Config | input | output | num-prompts (= concurrency) | Regime |
| --- | --- | --- | --- | --- |
| 1 — `3338/128/16` | 3338 | 128 | 16 | long prefill (near `--max-model-len`) |
| 2 — `128/128/16` | 128 | 128 | 16 | balanced (short prefill) |

Run both; these two are the default spec. Use the user's values if they specify different lengths. Keep `input + output <= --max-model-len` (4096).

Confirm the endpoint serves (a healthy `/v1/models` alone does not prove generation works):

```bash
PORT=<your-port>
curl -sf http://localhost:$PORT/health                                # {"status":"ok"}
curl -sf http://localhost:$PORT/v1/models | grep -o '"id":"[^"]*"'    # the served-model-name
curl -sf http://localhost:$PORT/v1/completions -H 'Content-Type: application/json' \
  -d "{\"model\":\"<served-model-name>\",\"prompt\":\"ping\",\"max_tokens\":4,\"temperature\":0}" >/dev/null && echo gen OK
```

Drive the workload with **`tests/bench_serving.py` (default — no install)**. The repo's own async benchmark drives the endpoint with aiohttp (already in the env) and prints TTFT, per-token decode interval, throughput (req/s, tok/s), and latency p50/p99. Its `--input-len` builds a fixed-length synthetic prompt. Run both configs:

```bash
PORT=<your-port>
# config 1 (long prefill)
python tests/bench_serving.py --host localhost --port "$PORT" \
    --input-len 3338 --max-tokens 128 -n 16 -c 16 --stream
# config 2 (balanced)
python tests/bench_serving.py --host localhost --port "$PORT" \
    --input-len 128 --max-tokens 128 -n 16 -c 16 --stream
```

`--stream` enables TTFT + per-token decode measurement. `-n` is the request count, `-c` the concurrency. `--input-len` is approximate (repeated text); the exact token count shows up in the server log and in the SA_PROFILE `prefill_fwd` span.

**Alternative: `vllm bench serve` (optional, needs vllm installed).** If vllm is available, the `vllm-bench-perf` skill's `vllm bench serve` against the same endpoint is an equivalent driver with its own TTFT/TPOT/throughput report — same two configs map to `--random-input-len` / `--random-output-len` / `--num-prompts`:

```bash
vllm bench serve --backend vllm --model <served-model-name> \
  --base-url http://localhost:$PORT --endpoint /v1/completions --dataset-name random \
  --random-input-len 3338 --random-output-len 128 --num-prompts 16 --request-rate inf   # config 1
```

Either driver works; prefer `tests/bench_serving.py` (repo-native, no install).

## 5. Flush and merge fragments

The recorder writes fragments with default buffering and flushes on process close. To get a complete merged `trace.json`:

- **Preferred:** stop the server gracefully (`SIGTERM`/`SIGINT` to the `pypto-serving` process). The application-shutdown path waits for the worker and calls `merge_profile()`, producing `<SA_PROFILE_OUTPUT>/trace.json`.
- **If the server was killed ungracefully:** run `./scripts/merge_profile.sh <SA_PROFILE_OUTPUT>` (or `SA_PROFILE_OUTPUT=<dir> ./scripts/merge_profile.sh`). Stop all profiled processes first so buffered events are flushed.

Fragments are retained after merging, so aggregation also works directly on `fragments/trace.*.jsonl` without a merged file — useful for an interim read while the server is still running (some recent events may still be buffered).

## 6. Read operator timing from the trace

Each line in a fragment is one Chrome trace event. Duration events are `{"ph":"X","name":...,"cat":...,"ts":...,"dur":<us>,"pid":...,"tid":...,"args":{...}}`. To get operator timing, filter `ph=="X"`, group by `name`, and sum `dur` (microseconds); sort by total descending.

Categories and the kernel names that matter most:

| `cat` | Representative `name` | What it measures |
| --- | --- | --- |
| `kernel` | `kernel.prefill_fwd`, `kernel.decode_fwd`, `kernel.greedy_sample_fwd` | NPU kernel dispatch per prefill / decode-step / sampling. **The operator breakdown lives here.** |
| `kernel` | `<name>.worker_run` | The inner `worker.run` device dispatch (a sub-span of the above). |
| `scheduler` | `scheduler.schedule`, `scheduler.wait_worker_output`, `scheduler.process_step_output` | Scheduling and the host wait for the device each step. |
| `worker` | `WorkerProcess.execute_step`, `WorkerProcess.batch_prefill`, `WorkerProcess.batch_decode` | Worker-side step/prefill/decode wrappers. |
| `executor` | `PyptoExecutor.run_prefill`, `PyptoExecutor.run_decode` | Executor entry into the kernels. |
| `request` | `http.completions`, `http.stream_completion` | End-to-end per-request latency. |

For kernel events, prefer the **device-side** time carried in `args`: `device_wall_us` (device run time) and `host_wall_us` (host-side wall time), added by the runner after each dispatch. These are more accurate than the span `dur`, which includes host dispatch overhead. Aggregating `args.device_wall_us` by kernel name gives the true device cost.

Compact aggregator (writes nothing into the repo; run from anywhere):

```python
import glob, json, os
from collections import defaultdict
D="<SA_PROFILE_OUTPUT>"; tot=defaultdict(lambda:[0.0,0,"",0.0,0])  # dur,count,cat,dev_us,dev_n
def events():
    m=os.path.join(D,"trace.json")
    if os.path.isfile(m):
        for e in json.load(open(m)).get("traceEvents",[]): yield e
    for f in sorted(glob.glob(os.path.join(D,"fragments","trace.*.jsonl"))):
        for line in open(f):
            line=line.strip()
            if line:
                try: yield json.loads(line)
                except json.JSONDecodeError: pass
for e in events():
    if e.get("ph")!="X": continue
    r=tot[e.get("name","?")]; r[0]+=float(e.get("dur",0) or 0); r[1]+=1; r[2]=e.get("cat","")
    a=e.get("args") or {}
    if isinstance(a,dict) and a.get("device_wall_us") is not None:
        r[3]+=float(a["device_wall_us"]); r[4]+=1
for name,(d,n,c,dev,dn) in sorted(tot.items(),key=lambda kv:kv[1][0],reverse=True)[:25]:
    print(f"{d/1000:10.3f}ms  n={n:<5} {c:<10} {name}")
```

**Exclude startup spans** when analyzing the request period: `AsyncLLMEngine.start`, `WorkerProcess.init_device_and_model`, `PyptoExecutor.register_model`, `Qwen314BModelRunner.prepare_l3_worker`, and `upload_static_tensors` are one-time and dominate totals if included.

## 7. Interpreting the results

- `kernel.decode_fwd` fires **once per decode iteration** (batched across all in-flight sequences), so its mean ≈ per-step decode time ≈ **TPOT**, and `count` ≈ total decode steps. `kernel.prefill_fwd` fires once per prefill; its mean ≈ per-prefill cost and dominates **TTFT**. `kernel.greedy_sample_fwd` is cheap (a few ms).
- Compare `args.device_wall_us` to the span `dur`: a large gap means host dispatch/scheduling overhead, not device compute.
- `scheduler.wait_worker_output` ≈ `kernel.decode_fwd` means the host is blocked on the device each step (little host idle). If `wait_worker_output` ≫ `decode_fwd`, the host is stalling elsewhere.
- Decode throughput ≈ `1000 / mean(kernel.decode_fwd ms)` requests·tokens per second per batch; multiply by the running batch size for aggregate token throughput.

## 8. Troubleshooting

**a) Warmup crashes with `AICore error 507018` / `bounded device drain failed` / `Worker reported invalid KV cache page count: 0`.** A known NPU device-drain flake during the warmup prefill dispatch, not a profiling or config problem. The card is force-reset automatically. Retry the same command.

**b) `[Errno 98] address already in use` on bind.** The default port 8000 is taken (often by a teammate's server). Re-run with `--port <other>` and point the workload at the new port.

**c) No `kernel.*_fwd` events, only `e2e`/scheduler spans.** `SA_PROFILE_LEVEL` does not include `kernel`. Set it to `verbose` or `e2e,kernel`, and confirm the env is exported **before** launching the server so the worker process inherits it.

**d) `trace.json` is missing or much smaller than the fragments.** The server was killed ungracefully so the merge did not run. Stop all profiled processes, then run `./scripts/merge_profile.sh <SA_PROFILE_OUTPUT>`, or aggregate the fragments directly.

**e) Every request returns 422.** The `model` field does not match the served-model-name from `/v1/models`, or `prompt_tokens + max_tokens > --max-model-len`. A single isolated 422 under concurrency is usually transient — retry that request.

**f) Stale results from a previous run.** Reusing `SA_PROFILE_OUTPUT` is fine (a new main process clears old fragments), but a leftover `trace.json` from before the clear can confuse an interim read. Use a fresh path per run, or merge after the new run completes.

## 9. Checklist

1. Pick a fresh absolute `SA_PROFILE_OUTPUT` and set `SA_PROFILE_LEVEL=verbose` (must include `kernel`).
2. Start `pypto-serving` (queue-wrapped if needed) with `--port` chosen to avoid collisions; wait for `Application startup complete`.
3. Confirm `/health`, the served-model-name from `/v1/models`, and that one `/v1/completions` returns a completion.
4. Drive the workload with `tests/bench_serving.py` (default, no install): run both spec configs `3338/128/16` and `128/128/16` via `--input-len` / `--max-tokens` / `-n` / `-c` / `--stream`. `vllm bench serve` is an optional alternative.
5. Stop the server gracefully (or run `scripts/merge_profile.sh`) to produce `trace.json`.
6. Aggregate `ph=X` spans by `name`; for kernel rows prefer `args.device_wall_us`; exclude the one-time startup spans.
7. Report: per-kernel total/count/mean (prefill vs decode vs sample), TPOT ≈ mean `decode_fwd`, the device-vs-host gap, and the port/model/workload actually used.
