---
name: qwen3-14b-offline-perf-test
description: Run one Qwen3-14B offline generation or decode performance test in the pypto-serving repository and summarize the actual versions, scenario, TPOT/throughput, timing breakdown, calculation method, and whether prefill plus device-side embedding/sampling ran.
---

# Qwen3-14B Offline Perf Test

Use this skill to run one repository-local Qwen3-14B offline performance test and report the result in a compact, comparable format.

Do not hard-code a specific commit, server name, user directory, or one-off experiment path. Test the checkout and dependencies requested by the user, then print what actually ran.

## Scope

Prefer the serving repository's offline generate mode (same engine as serving, no HTTP server):

```bash
pypto-serving --model /path/to/Qwen3-14B --prompt "..." --generate-config '{"max_new_tokens": 128}'
```

Repeat `--prompt` for a batch (each occurrence is one request, scheduled with continuous batching). Do not use HTTP/FastAPI/Uvicorn serving unless the user explicitly asks for that path.

Keep the normal serving-core flow when possible: tokenize, prefill, KVCache, decode, sampling, and detokenize. If prefill cannot run, clearly label the result as `cached-prefill decode`, `serving-core probe`, or `decode-only`.

## Scenario

Use user-provided values first.

| Item | Default |
| --- | --- |
| Model | Qwen3-14B |
| Prompt | user-provided prompt file or prompt text |
| Batch size | ask the user; if absent for a decode test, use batch size 16 |
| Max sequence length | 4096 |
| Decode tokens | 128 |
| Sampling | greedy, `temperature=0` or equivalent |
| Run count | 1 |

If the user has not provided a prompt path or prompt content, ask the user to provide the prompt storage location before running the benchmark. In the final result, always state the actual prompt path or source, prompt token count if available, batch size, max sequence length, decode token count, sampling mode, and device id.

## Version Audit

Before running, save a short version audit next to the log. Report actual code versions, not intended versions.

Recommended command:

```bash
RUN_DIR="${RUN_DIR:-artifacts/qwen3_14b_offline_perf_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

{
  echo "== serving =="
  pwd
  git rev-parse HEAD
  git status --short

  echo
  echo "== dependency repositories =="
  for d in pypto-lib pypto-main pypto simpler runtime pto-isa; do
    if [ -d "$d/.git" ] || git -C "$d" rev-parse --git-dir >/dev/null 2>&1; then
      echo "-- $d --"
      git -C "$d" rev-parse HEAD
      git -C "$d" status --short
    fi
  done
} 2>&1 | tee "$RUN_DIR/version_audit.log"
```

If a dependency version cannot be detected from a checkout or package version, write `unknown`; do not guess.

## Run Offline Test

Use a real prompt file when possible. If the entry supports `--prompt-file`, prefer it. If it only supports `--prompt`, pass the file content as text.

If the server requires a queue wrapper such as `task-submit`, collect the queue command and required environment variables first, then run the same direct command inside that wrapper. Keep the direct command and arguments visible in the log.

Single-run template:

```bash
MODEL_DIR=/path/to/Qwen3-14B
PROMPT_FILE=/path/to/prompt.txt
DEVICE_ID=${DEVICE_ID:-0}
BATCH_SIZE=${BATCH_SIZE:-16}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
RUN_DIR="${RUN_DIR:-artifacts/qwen3_14b_offline_perf_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"

sha256sum "$PROMPT_FILE" | tee "$RUN_DIR/prompt.sha256"
wc -c "$PROMPT_FILE" | tee "$RUN_DIR/prompt.bytes"
PROMPT_TEXT="$(cat "$PROMPT_FILE")"

# One --prompt occurrence per request: BATCH_SIZE=16 gives a real batch of 16.
PROMPT_ARGS=()
for _ in $(seq "$BATCH_SIZE"); do PROMPT_ARGS+=(--prompt "$PROMPT_TEXT"); done

pypto-serving \
  --model "$MODEL_DIR" \
  --served-model-name "qwen3-14b-offline" \
  --platform "${PLATFORM:-a2a3}" \
  --device "$DEVICE_ID" \
  --max-model-len "$MAX_SEQ_LEN" \
  --max-num-seqs "$BATCH_SIZE" \
  --generate-config '{"max_new_tokens": '"$MAX_NEW_TOKENS"', "temperature": 0, "top_p": 1}' \
  "${PROMPT_ARGS[@]}" \
  --profile \
  2>&1 | tee "$RUN_DIR/run.log"
```

`--max-num-seqs` sets the scheduler batch capacity; the actual batch is the
number of `--prompt` occurrences. The entry prints an aggregate tokens/s
summary; per-phase (prefill vs decode) and per-kernel splits come from the
`--profile` trace or the STRACE method below.

## Runtime Evidence

Briefly confirm whether device-side embedding/sampling is enabled and whether real prefill ran successfully. If either capability is unavailable or the log does not expose enough evidence, mark it as `not provided by this version/log`.

For prefill, report prefill time or TTFT and the prompt token count if available. If prefill fails, do not call the result full offline generation.

## Metric Lookup And Calculation

Use the source guidance below to know where to read each metric while analyzing the logs. Do not print a per-metric source column in the final answer by default; only mention the source when the value is derived, ambiguous, missing, or the user asks where it came from. If this version does not print a metric, write `not provided`.

Report sampling location as `device`, `host`, or `not provided`. Sampling location can affect decode token accounting; for example, if prefill already returns the first sampled token, `run_decode` calls may be one fewer than `max_new_tokens`.

Use these calculation rules:

| Metric | Source / calculation |
| --- | --- |
| TPOT | Prefer the value printed by the offline entry. If derived, use decode wall time divided by the number of decode output tokens, and state the denominator. |
| Throughput | Prefer the printed decode throughput. If derived, use `1000 / TPOT_ms` or `decode_tokens / decode_seconds`, and state which formula was used. |
| run_decode | Use the printed decode wall time or call count. If missing, write `not provided`. |
| simpler_run | Use STRACE/full tree per-decode-step average. State whether the average uses all decode steps or skips warmup steps. |
| runner_run | Use STRACE/full tree child timing under `simpler_run`. If unavailable, write `not provided`. |
| device_wall | Use runtime/device timing from STRACE/full tree. If unavailable, write `not provided`. |
| Effective | Prefer an explicit `Effective` field if the log prints one. Otherwise use STRACE/full tree `simpler_run.runner_run.device_wall.graph_build` as `Effective ~= graph_build`, averaged over decode steps. If neither exists, write `not provided`. If using `device_wall [dev]` as a coarse proxy, label it separately and do not mix it with the graph_build proxy. |
| args | Use STRACE/full tree host argument preparation/staging time. |
| validate | Use STRACE/full tree runtime validation/checking time. |
| prebuilt | Use STRACE/full tree prebuilt callable/program reuse overhead. |
| inter-step gap | Derive as `TPOT - simpler_run` only when both are available and use the same token/step denominator. |

Use these relationships only as a sanity check, not as exact equations:

```text
TPOT ~= simpler_run + inter-step gap
simpler_run ~= args + prebuilt + runner_run + validate
runner_run ~= device_wall + host scheduling overhead
device_wall ~= Effective + small device-side phases
Effective ~= main NPU model compute
```

## Final Output

Answer in Chinese and keep it concise.

Include:

1. Version information: serving, pypto-lib, pypto, simpler/runtime, pto-isa, and any `unknown` entries.
2. Key metrics: TPOT, throughput, run_decode, simpler_run, runner_run, device_wall, Effective, args, validate, prebuilt, inter-step gap. Keep the table focused on metric names and values; include source notes only for derived, ambiguous, or missing values.
3. Simple metric explanation: explain in one or two lines where the time mainly goes and how TPOT/throughput should be read, and include these timing relationships:
   ```text
   TPOT ~= simpler_run + inter-step gap
   simpler_run ~= args + prebuilt + runner_run + validate
   runner_run ~= device_wall + host scheduling overhead
   device_wall ~= Effective + small device-side phases
   Effective ~= main NPU model compute
   ```
4. Brief scenario: model, prompt path/token count, batch size, max sequence length, decode tokens, sampling mode and sampling location, device id, whether prefill really ran, and whether the result is full offline generation or a fallback mode.
