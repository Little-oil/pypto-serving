---
name: qwen3-14b-offline-op-timing
description: Test and analyze OFFLINE generation performance of Qwen3-14B via the `pypto-serving --prompt` generate mode, using the built-in SA_PROFILE Chrome-trace recorder (--profile CLI options) for a per-operator / per-kernel time breakdown (prefill vs decode, device-side kernel duration). The offline counterpart of qwen3-14b-online-perf-test. Use when the user wants to profile a single offline generation for operator/kernel timing (not the HTTP server), or compare offline vs online kernel costs. Uses the same PTO2_* runtime env and the same two workload configs (3338/128/16 and 128/128/16). For the STRACE / host-side offline method see qwen3-14b-offline-perf-test; for the online HTTP server see qwen3-14b-online-perf-test.
---

# Qwen3-14B offline generation performance profiling

Offline counterpart of `qwen3-14b-online-perf-test`. Same profiling mechanism (the built-in `SA_PROFILE` Chrome-trace recorder in `pypto_serving.tools.profile`) and same `PTO2_*` runtime env, but it profiles a **single offline generation** through the `pypto-serving --prompt` generate mode (same engine as serving: scheduler + worker process, no HTTP server / vllm bench / endpoint involved).

The trace it produces has the **same schema and kernel names** as the online skill, so reading and aggregating it is identical — see `qwen3-14b-online-perf-test` §6–§7 for the event schema, kernel-name table, aggregator script, and interpretation. Only the run command and the workload mapping differ (below).

Do not hard-code a commit, user, device id, or path. Use the user's model dir, output path, device, and prompt source.

---

## 1. Prerequisites

- A Conda environment with `pypto-serving` installed (the CLI is the offline entry); `transformers` is available for the tokenizer helper.
- The model weights at a local model dir (e.g. Qwen3-14B).
- An NPU device; run queue-wrapped with `task-submit` if the box requires it — `--device auto` fills the `{}` in `--device {}`.

## 2. Enable profiling

Configured with the CLI profile options (the offline entry no longer reads `SA_PROFILE_*`): `--profile` plus `--profile-output <absolute dir, fresh per run>` and `--profile-level verbose` (must include `kernel`, or there are no `kernel.*_fwd` spans). Output layout is `fragments/trace.<pid>.jsonl` plus a merged `trace.json`. Keep whatever `PTO2_*` runtime env the run template uses.

## 3. Run the offline generation

The generate mode takes `--prompt` as **required text, repeatable** (each occurrence is one request; there is no `--prompt-len` / `--prompt-file` / synthetic option), so for a fixed input length build a prompt of exactly N tokens with the tokenizer first. Template (queue-wrapped; `--device {}` is filled by `--device auto`):

```bash
# helper: emit a prompt of exactly N tokens (repeated single-token word; "hello"
# is 1 token for the Qwen3 tokenizer, so "hello"*N is exactly N tokens — no
# decode/encode round-trip loss)
make_prompt() { python -c "import sys; sys.stdout.write('hello'*int(sys.argv[1]))" "$1"; }

MODEL_DIR=/path/to/Qwen3-14B
PROMPT="$(make_prompt 3338)"      # N = the config's input length

# 16 identical --prompt occurrences = one real batch of 16 concurrent requests.
# The flags defer $PROMPT to the run shell, so export it (and MODEL_DIR).
PROMPT_FLAGS=""
for _ in $(seq 16); do PROMPT_FLAGS+=' --prompt "$PROMPT"'; done
export MODEL_DIR PROMPT

task-submit --device auto --run --max-time 0 --timeout 0 \
"PTO2_OP_EXECUTE_TIMEOUT_US=50000000 PTO2_STREAM_SYNC_TIMEOUT_MS=55000 \
PTO2_RING_HEAP=2147483648 PTO2_RING_TASK_WINDOW=262144 PTO2_RING_DEP_POOL=262144 \
pypto-serving \
    --model \"$MODEL_DIR\" \
    $PROMPT_FLAGS \
    --platform a2a3 \
    --device {} \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 4096 \
    --no-enable-prefix-caching \
    --npu-memory-utilization 0.9 \
    --generate-config '{\"max_new_tokens\": 128}' \
    --profile \
    --profile-output /abs/path/profile-offline \
    --profile-level verbose"
```

Notes specific to the offline entry:

- Sixteen `--prompt` occurrences run through `engine.generate_batch` — a **real batch** of 16 concurrent requests (each identical, matching the workload spec). `--max-num-seqs 16` sets the scheduler batch capacity; the actual batch is the number of `--prompt` occurrences.
- `make_prompt N` builds an EXACT N-token prompt by repeating `"hello"` (1 token for the Qwen3 tokenizer). Unlike sentence-repeat + decode, no decode/encode round-trip loss.
- When total prompt tokens across all requests exceed `--max-num-batched-tokens` (default 4096), the scheduler automatically splits into **chunked prefill**. Each prefill step packs up to 4096 new tokens (`--long-prefill-token-threshold` bounds per-request chunk length). For config 1 (`3338/128/16`, 16×3338≈53k total), this produces many prefill steps before decode begins.
- `--profile` wraps the generation window with profile start/stop and merges the trace when the run finishes. The CLI prints an aggregate tokens/s summary; the per-operator / per-kernel breakdown comes from the trace.
- The prompt is injected into the `task-submit` quoted string via `\"$PROMPT\"`; the tokenizer-built prompt is plain text with no shell-special characters. If you supply your own prompt, make sure it is shell-safe or pass it the same way.

## 4. Workload spec — two configs

The same two configs as the online skill. Offline they map to generate-mode args as `prompt-tokens / --generate-config max_new_tokens / --max-num-seqs`:

| Config | prompt tokens (`make_prompt N`) | `max_new_tokens` | `--max-num-seqs` | Regime |
| --- | --- | --- | --- | --- |
| 1 — `3338/128/16` | 3338 | 128 | 16 | long prefill |
| 2 — `128/128/16` | 128 | 128 | 16 | balanced |

Run both (one batched generation each; rebuild the prompt with the matching `N`). Keep `prompt_tokens + max_new_tokens <= --max-model-len` (4096).

## 5. Flush and merge fragments

Automatic for offline: the CLI calls `stop_profile` + `merge_profile()` when the generation run finishes, producing `<--profile-output>/trace.json`. If it was killed before that, run `./scripts/merge_profile.sh <profile-output-dir>` (stop the process first so buffered events flush). Fragments can also be aggregated directly without a merged file.

## 6. Read operator timing from the trace

Identical to the online skill — the trace is the same Chrome trace-event format with the same kernel names. Use the event schema, the kernel-name table, and the compact aggregator in `qwen3-14b-online-perf-test` §6 (filter `ph=="X"`, group by `name`, sum `dur`; for kernel rows prefer `args.device_wall_us`; exclude one-time startup spans).

## 7. Interpreting the results

Same metrics as online §7 (batching works identically: 16 `--prompt` occurrences generate a real batch).

## 8. Troubleshooting

**a) Warmup crashes with `AICore error 507018` / `bounded device drain failed`.** The known NPU device-drain flake (same as online §8). The card auto-resets; retry the same command.

**b) No `kernel.*_fwd` events, only `e2e` spans.** `--profile-level` does not include `kernel`. Set it to `verbose` (or `e2e,kernel`) before launching.

**c) `trace.json` is missing or tiny.** The process was killed before the merge. Stop it, then run `./scripts/merge_profile.sh <profile-output-dir>`, or aggregate the fragments directly.

**d) Prompt length mismatch.** The generate mode has no length knob — the input length is the token count of each `--prompt`. Use `make_prompt N` to hit an exact length; verify with the tokenizer if precision matters. Keep `prompt_tokens + max_new_tokens <= --max-model-len`.

**e) Not actually batched.** The command must include sixteen `--prompt` occurrences (not just `--max-num-seqs 16`) for real batch-16 generation. With a single `--prompt`, the run is 1 request regardless of `--max-num-seqs`.

## 9. Checklist

1. Pick a fresh absolute `--profile-output` and set `--profile-level verbose` (must include `kernel`).
2. Build the prompt to the config's token length with `make_prompt N`.
3. Run the generate mode (queue-wrapped) with the config's `--generate-config max_new_tokens` and `--max-num-seqs`, one `--prompt` per request; wait for the generation to finish.
4. Repeat for the second config (`3338/128/16` and `128/128/16`).
5. Confirm the CLI merged the trace (or run `scripts/merge_profile.sh`).
6. Aggregate `ph=X` spans by `name` using the online §6 aggregator; report `kernel.prefill_fwd` / `decode_fwd` / `greedy_sample_fwd` total+count+mean, `args.device_wall_us`, and TPOT ≈ mean `decode_fwd`.
