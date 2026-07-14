---
name: vllm-bench-perf
description: Measure serving and offline performance with the vLLM `vllm bench` CLI using the synthetic `random` dataset, which requires no dataset download. Covers the `serve` subcommand (online TTFT, TPOT, and throughput) and the `throughput` and `latency` subcommands (offline). Use when the user requests benchmarking of throughput, latency, or TTFT, or a stress or performance test. Provides a complete from-scratch workflow covering installation, server startup, and benchmark execution.
---

# vLLM performance benchmarking (`vllm bench`)

The `vllm bench` CLI measures serving latency and throughput. It provides the subcommands `serve` (an online client that targets an endpoint), `throughput` (offline, using its own engine), and `latency` (offline). This document describes a complete from-scratch workflow and uses the synthetic `random` dataset, so no dataset download is required.

---

## 1. Prerequisites

- **Python 3.9 or later** (3.10 or 3.11 recommended).
- For the **`serve`** subcommand (online client): only the vllm package and a reachable OpenAI-compatible endpoint are required; no accelerator is needed on the client.
- For the **`throughput`** and **`latency`** subcommands (offline): vLLM starts its own engine, which requires a **vLLM-supported accelerator** (an NVIDIA or AMD GPU) and the model weights. If the accelerator is not supported by upstream vLLM, use `serve` against an available OpenAI-compatible endpoint instead.

## 2. Installation (from scratch)

`vllm bench` is included with the vllm package; no separate benchmark wheel exists:

```shell
conda create -n vllm python=3.11 -y
conda activate vllm
pip install vllm
vllm bench serve --help        # confirm that the CLI is available
```

The full `vllm` installation downloads torch and the CUDA or ROCm dependencies, which occupy several gigabytes. A lighter alternative that supports only the `serve` client is the repository script `benchmarks/benchmark_serving.py`, which requires only httpx and aiohttp.

## 3. Start a vLLM server (required for `vllm bench serve`)

The `serve` subcommand is a client and therefore requires a running OpenAI-compatible endpoint. Start one with vLLM, substituting your own model and port:

```shell
MODEL=<your-model>            # for example, meta-llama/Llama-3.1-8B-Instruct, or a local path
PORT=<your-port>              # for example, 8000
vllm serve "$MODEL" --port "$PORT" --served-model-name "$MODEL"
```

Wait for the log line `INFO: Application startup complete.` or `Uvicorn running on http://0.0.0.0:<port>` before running the benchmark.

## 4. `vllm bench serve` — online serving performance (recommended, `random` dataset)

The `random` dataset generates synthetic prompts in process, with fully controllable input and output lengths and no download:

```bash
# random-input-len / random-output-len : per-request token lengths (prefill-bound versus decode-bound regime)
# num-prompts          : number of requests
# request-rate inf     : maximum-throughput stress test (default); use a finite value for Poisson arrivals
vllm bench serve \
  --backend vllm \
  --model "$MODEL" \
  --base-url http://localhost:$PORT \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 200 \
  --request-rate inf
```

**Load-pattern flags** (these control how traffic is generated):
- `--request-rate` — the target request rate in requests per second. The default value, `inf`, issues all requests immediately, producing a maximum-throughput stress test; a finite value produces Poisson arrivals.
- `--burstiness` — the Gamma distribution shape parameter. A value of 1.0 produces Poisson traffic; values below 1.0 produce bursty traffic; values above 1.0 produce uniform traffic. This flag takes effect only when `--request-rate` is finite.
- `--max-concurrency` — a cap on the number of concurrent in-flight requests, used to simulate gateway backpressure.
- Ramp-up: `--ramp-up-strategy linear|exponential --ramp-up-start-rps … --ramp-up-end-rps …`.
- Sampling: `--temperature`, `--top-p`, and `--top-k`.

Options specific to the `random` dataset: `--ignore-eos` forces generation of exactly `--random-output-len` tokens (otherwise generation stops at the end-of-sequence token), and `--tokenizer <path>` is required when the model identifier is not a loadable local tokenizer.

**Saving and visualizing results.** Use `--save-result --save-detailed --result-dir ./out` to persist the results, and use `--plot-timeline --plot-dataset-stats` to generate HTML charts.

**Reported metrics.** Successful requests; benchmark duration; request throughput (req/s); output token throughput (tok/s); total token throughput; **TTFT** (mean, median, and p99); **TPOT** (mean, median, and p99); and **ITL** (mean, median, and p99).

## 5. `vllm bench throughput` — offline throughput

This subcommand starts its own vLLM engine and therefore requires vLLM, a supported accelerator, and the model weights:

```bash
vllm bench throughput \
  --model "$MODEL" \
  --dataset-name random \
  --random-input-len 1024 --random-output-len 128 \
  --num-prompts 1000
```

Result line: `Throughput: X requests/s, Y total tokens/s, Z output tokens/s`.

## 6. `vllm bench latency` — offline latency

This subcommand measures single-request or batch latency and starts its own engine. For the full set of options, run `vllm bench latency --model "$MODEL" --help`.

## 7. Interpreting the results

- **TTFT** (time to first token) — the latency before the first token is produced; it is the dominant component of perceived responsiveness.
- **TPOT** (time per output token, excluding the first token) — the per-token decode time.
- **ITL** (inter-token latency) — the per-token latency, including variance.
- **Throughput** — `req/s` reflects end-to-end capacity, and output `tok/s` reflects generation bandwidth.

Load-pattern recommendations by objective:

| Objective | `--request-rate` | `--max-concurrency` | `--burstiness` |
|---|---|---|---|
| Maximum throughput (most common) | `inf` | a finite cap | not applicable |
| Realistic Poisson traffic | finite (5–20) | unset | 1.0 |
| Bursty or stress testing | high finite | unset | 0.1–0.5 |
| Latency profiling | low finite | unset | 2.0–5.0 |

Sweep `--random-input-len` and `--random-output-len` to compare prefill-bound and decode-bound regimes. Sweep `--request-rate` and `--max-concurrency` to identify the saturation point, then reduce the load by approximately 10 to 20 percent to obtain a stable measurement. Always report the **p99** value alongside the mean.

## 8. Selection and caveats

- To measure a **running endpoint**, use `serve`, which acts as a client against any OpenAI-compatible endpoint.
- To measure **offline engine** throughput or latency, use `throughput` or `latency`; each starts its own engine and requires a supported accelerator.
- `--request-rate inf --max-concurrency <N>` is the standard maximum-throughput configuration.
- Backends and endpoints: use `--backend vllm` with `/v1/completions` for raw completions, and `--backend openai-chat` with `/v1/chat/completions` for chat and multimodal requests.
- The `random` dataset is sufficient for performance testing. The remaining datasets (`sharegpt`, `sonnet`, `hf`, and `custom`) model realistic traffic shapes and are optional.
