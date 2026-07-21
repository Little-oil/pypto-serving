---
name: qwen3-14b-offline-op-timing
description: Test and analyze OFFLINE generation performance of Qwen3-14B via the npu_generate.py entry, using the built-in SA_PROFILE Chrome-trace recorder for a per-operator / per-kernel time breakdown (prefill vs decode, device-side kernel duration). The offline counterpart of qwen3-14b-online-perf-test. Use when the user wants to profile a single offline generation for operator/kernel timing (not the HTTP server), or compare offline vs online kernel costs. Uses the same PTO2_* + SA_PROFILE_* env and the same two workload configs (3338/128/16 and 128/128/16). For the STRACE / host-side offline method see qwen3-14b-offline-perf-test; for the online HTTP server see qwen3-14b-online-perf-test.
---

# Qwen3-14B offline generation performance profiling

Offline counterpart of `qwen3-14b-online-perf-test`. Same profiling mechanism (the built-in `SA_PROFILE` Chrome-trace recorder in `pypto_serving.tools.profile`), same `PTO2_*` runtime env, and the same two workload configs — but it profiles a **single offline generation** through `examples/model/qwen3_14b/npu_generate.py` instead of the HTTP server, so there is no vllm bench / endpoint involved.

The trace it produces has the **same schema and kernel names** as the online skill, so reading and aggregating it is identical — see `qwen3-14b-online-perf-test` §6–§7 for the event schema, kernel-name table, aggregator script, and interpretation. Only the run command and the workload mapping differ (below).

Do not hard-code a commit, user, device id, or path. Use the user's model dir, output path, device, and prompt source.

---

## 1. Prerequisites

- A Conda environment with `pypto-serving` installed (npu_generate.py imports `pypto_serving`); `transformers` is available for the tokenizer helper.
- The model weights at a local model dir (e.g. Qwen3-14B).
- An NPU device; run queue-wrapped with `task-submit` if the box requires it — `--device auto` fills the `{}` in `--device-id {}`.

## 2. Enable profiling

Same as online: set `SA_PROFILE_OUTPUT=<absolute dir, fresh per run>` and `SA_PROFILE_LEVEL=verbose` (must include `kernel`, or there are no `kernel.*_fwd` spans). Output layout is `fragments/trace.<pid>.jsonl` plus a merged `trace.json`. Keep whatever `PTO2_*` runtime env the run template uses.

## 3. Run the offline generation

`npu_generate.py` takes `--prompt` as **required text** (there is no `--prompt-len` / `--prompt-file` / synthetic option), so for a fixed input length build a prompt of exactly N tokens with the tokenizer first. Template (queue-wrapped; `--device-id {}` is filled by `--device auto`):

```bash
# helper: emit a prompt of exactly N tokens (repeated single-token word; "hello"
# is 1 token for the Qwen3 tokenizer, so "hello"*N is exactly N tokens — no
# decode/encode round-trip loss)
make_prompt() { python -c "import sys; sys.stdout.write('hello'*int(sys.argv[1]))" "$1"; }

MODEL_DIR=/path/to/Qwen3-14B
PROMPT="$(make_prompt 3338)"      # N = the config's input length

task-submit --device auto --run --max-time 0 --timeout 0 \
"SA_PROFILE_OUTPUT=/abs/path/profile-offline \
SA_PROFILE_LEVEL=verbose \
PTO2_OP_EXECUTE_TIMEOUT_US=50000000 PTO2_STREAM_SYNC_TIMEOUT_MS=55000 \
PTO2_RING_HEAP=2147483648 PTO2_RING_TASK_WINDOW=262144 PTO2_RING_DEP_POOL=262144 \
python examples/model/qwen3_14b/npu_generate.py \
    --model-dir \"$MODEL_DIR\" \
    --prompt \"$PROMPT\" \
    --platform a2a3 \
    --device-id {} \
    --max-seq-len 4096 \
    --max-new-tokens 128 \
    --max-num-seqs 16 \
    --num-prompts 16 \
    --max-num-batched-tokens 4096 \
    --no-enable-prefix-caching \
    --npu-memory-utilization 0.9"
```

Notes specific to the offline entry:

- `--num-prompts 16` calls `engine.generate_batch(model_id, [prompt]*16, config)` — **real batch** of 16 concurrent requests (each identical, matching the workload spec). `--max-num-seqs 16` sets the compile-time batch capacity.
- `make_prompt N` builds an EXACT N-token prompt by repeating `"hello"` (1 token for the Qwen3 tokenizer). Unlike sentence-repeat + decode, no decode/encode round-trip loss.
- When total prompt tokens across all requests exceed `--max-num-batched-tokens` (default 4096), the engine automatically splits into **chunked prefill**. Each `run_prefill` call packs up to 4096 new tokens, greedily filling requests in order (req0 first, then req1, etc.). Completed requests drop out of subsequent chunks. The decode loop runs after all prefill chunks finish. For config 1 (`3338/128/16`, 16×3338≈53k total), this produces ~14 chunks.
- `--profile` / `--profile-verbose` are **optional and separate** from `SA_PROFILE`: they print npu_generate's own phase / per-kernel timing summary. The `SA_PROFILE` trace is collected from the env vars alone.
- The prompt is injected into the `task-submit` quoted string via `\"$PROMPT\"`; the tokenizer-built prompt is plain text with no shell-special characters. If you supply your own prompt, make sure it is shell-safe or pass it the same way.
- The entry calls `merge_profile()` in its `finally` block, so `trace.json` is produced even if generation raises.

## 4. Workload spec — two configs

The same two configs as the online skill. Offline they map to npu_generate args as `prompt-tokens / --max-new-tokens / --max-num-seqs`:

| Config | prompt tokens (`make_prompt N`) | `--max-new-tokens` | `--max-num-seqs` | Regime |
| --- | --- | --- | --- | --- |
| 1 — `3338/128/16` | 3338 | 128 | 16 | long prefill |
| 2 — `128/128/16` | 128 | 128 | 16 | balanced |

Run both (one batched generation each; rebuild the prompt with the matching `N`). Keep `prompt_tokens + max-new-tokens <= --max-seq-len` (4096).

## 5. Flush and merge fragments

Automatic for offline: `npu_generate.py` calls `merge_profile()` in its `finally` block, producing `<SA_PROFILE_OUTPUT>/trace.json`. If it was killed before `finally`, run `./scripts/merge_profile.sh <SA_PROFILE_OUTPUT>` (stop the process first so buffered events flush). Fragments can also be aggregated directly without a merged file.

## 6. Read operator timing from the trace

Identical to the online skill — the trace is the same Chrome trace-event format with the same kernel names. Use the event schema, the kernel-name table, and the compact aggregator in `qwen3-14b-online-perf-test` §6 (filter `ph=="X"`, group by `name`, sum `dur`; for kernel rows prefer `args.device_wall_us`; exclude one-time startup spans).

## 7. Interpreting the results

Same metrics as online §7 (batching works identically now that `--num-prompts 16` generates a real batch).

## 8. Troubleshooting

**a) Warmup crashes with `AICore error 507018` / `bounded device drain failed`.** The known NPU device-drain flake (same as online §8). The card auto-resets; retry the same command.

**b) No `kernel.*_fwd` events, only `e2e` spans.** `SA_PROFILE_LEVEL` does not include `kernel`. Set it to `verbose` (or `e2e,kernel`) before launching.

**c) `trace.json` is missing or tiny.** The process was killed before `finally`. Stop it, then run `./scripts/merge_profile.sh <SA_PROFILE_OUTPUT>`, or aggregate the fragments directly.

**d) Prompt length mismatch.** `npu_generate.py` has no length knob — the input length is the token count of `--prompt`. Use `make_prompt N` to hit an exact length; verify with the tokenizer if precision matters. Keep `prompt_tokens + max-new-tokens <= --max-seq-len`.

**e) Not actually batched.** The command must include `--num-prompts 16` (not just `--max-num-seqs 16`) for real batch-16 generation. Without `--num-prompts`, `npu_generate.py` runs only 1 request regardless of `--max-num-seqs`.

## 9. Checklist

1. Pick a fresh absolute `SA_PROFILE_OUTPUT` and set `SA_PROFILE_LEVEL=verbose` (must include `kernel`).
2. Build the prompt to the config's token length with `make_prompt N`.
3. Run `npu_generate.py` (queue-wrapped) with the config's `--max-new-tokens` and `--max-num-seqs`; wait for the generation to finish.
4. Repeat for the second config (`3338/128/16` and `128/128/16`).
5. Confirm `merge_profile()` ran in `finally` (or run `scripts/merge_profile.sh`).
6. Aggregate `ph=X` spans by `name` using the online §6 aggregator; report `kernel.prefill_fwd` / `decode_fwd` / `greedy_sample_fwd` total+count+mean, `args.device_wall_us`, and TPOT ≈ mean `decode_fwd`.
