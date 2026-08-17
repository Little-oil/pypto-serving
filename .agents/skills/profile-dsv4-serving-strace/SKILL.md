---
name: profile-dsv4-serving-strace
description: Run the fused one-L2 DeepSeek V4 online serving configuration for 20 output tokens on any eight assigned devices, capture Simpler host STRACE plus verbose serving/framework SA_PROFILE spans, and generate a combined Perfetto swimlane. Use for profiling DSV4 serving, DSV4 吐字20个, capturing Simpler host logs, inspecting serving overhead, or producing an 8-rank STRACE 泳道图 across local shells, containers, or scheduler-managed environments.
---

# Profile DSV4 Serving STRACE

Run one deterministic DSV4 MTP HTTP completion and turn the serving profile plus
Simpler host `[STRACE]` log into validated Perfetto artifacts, including one combined
swimlane aligned on the shared host monotonic clock.

## Acquire devices independently

Obtain exactly eight NPUs using the environment's normal resource mechanism. Run the
workflow inside that allocation. Do not assume a particular scheduler, host, device range,
model mount, Python environment, or PTOAS release.

The bundled runner intentionally has no dependency on `task-submit`, Slurm, Kubernetes, or
another site-specific launcher. A scheduler job may invoke the same command as its payload.

## Run the workflow

From the repository root:

```bash
bash .agents/skills/profile-dsv4-serving-strace/scripts/run_profile.sh \
  --model-dir /path/to/dsv4-flash-w8a8 \
  --use-compile-cache \
  --devices 0,1,2,3,4,5,6,7
```

The defaults are:

- request: prompt `Huawei is`, `temperature=0`, `max_tokens=20`
- serving: DP=8, EP=8, MTP enabled, fused one-L2 decode, prefix cache disabled
- Simpler: host STRACE enabled with `PYPTO_RUNTIME_LOG=v9`
- device STRACE: disabled with `SIMPLER_DEVICE_STRACE_ENABLE=0`
- runtime: 2 GiB ring heap, with task and dependency windows set to 131072
- timeouts: Simpler scheduler 320 s, op execution 400 s, stream sync 440 s, and
  serving worker step 1800 s
- serving profiling: on-demand `--profile --profile-level verbose`, with output under the
  artifact directory

Pass `--use-compile-cache` to reuse kernels from the serving worker's device-specific
`pypto_build_dir`. Reuse the same assigned devices, model configuration, and kernel sources;
the current cache has no fingerprinting and stale binaries are not detected automatically.

Use the current `SIMPLER_SCHEDULER_TIMEOUT_MS`, `SIMPLER_OP_EXECUTE_TIMEOUT_US`,
and `SIMPLER_STREAM_SYNC_TIMEOUT_MS` names. The former `PTO2_*TIMEOUT*` names are
not compatibility aliases and do not affect current Simpler runtime timeouts.

Select paths and assigned devices explicitly when defaults are unsuitable:

```bash
bash .agents/skills/profile-dsv4-serving-strace/scripts/run_profile.sh \
  --model-dir "$MODEL_DIR" \
  --python "$VIRTUAL_ENV/bin/python" \
  --devices "$ASSIGNED_DEVICE_IDS" \
  --artifact-dir artifacts/dsv4-profile-$(date +%Y%m%d-%H%M%S)
```

The wrapper also accepts `PYPTO_DSV4_MODEL_DIR`, `PYPTO_PROFILE_PYTHON`,
`PYPTO_PROFILE_DEVICES`, `PYPTO_PROFILE_RUN_ID`, and `PYPTO_USE_COMPILE_CACHE=1`.
Device discovery falls back to `TASK_DEVICE`, then `ASCEND_RT_VISIBLE_DEVICES`. If none
is set, pass `--devices` explicitly; the runner never assumes a device range.

If `simpler_setup` is not installed in the selected Python environment, set
`PYPTO_RUNTIME_ROOT` to a PyPTO runtime source directory containing `simpler_setup/`.

## Understand the two traces

The workflow produces two source profiling views and one combined view:

- `serving-trace/trace.json`: verbose `SA_PROFILE` spans captured between
  `POST /start_profile` and `POST /stop_profile`, covering HTTP, scheduler,
  worker/executor, preparation, MTP, and kernel wrapper paths.
- `simpler-swimlane.json`: Simpler host STRACE spans such as `simpler_run`, `bind`,
  `runner_run`, and `validate`.
- `serving-strace-swimlane.json`: all serving/framework spans plus eight detailed
  host STRACE rank lanes on the same host monotonic timeline.

Device STRACE is intentionally off. Therefore the supported input has no `clk=dev`,
`device_wall`, `orch`, `sched`, or Simpler Effective measurement. Report Effective as
unavailable, never as zero. The scripts reject device-STRACE and non-one-L2 inputs.

Serving child processes can write several complete `[STRACE]` records on one physical log
line. The analyzer splits at every marker before calling Simpler's built-in
`parse_spans`, `group_invocations`, `to_chrome_trace`, and `_round_metrics` APIs. Require
four prefill invocations followed by exactly one fused invocation per proposed decode step.

## Validate before reporting success

Require all of the following:

1. The runner exits with code 0.
2. `completion-response.json` reports the requested completion-token count (20 by default,
   or the value selected with `--max-tokens`).
3. `completion.txt` contains generated text.
4. `server.log` contains successful SA profiler start/stop, final request, and MTP
   acceptance lines.
5. Eight distinct `[chip_process pid=... dev=...] ready` mappings are present.
6. `serving-trace/trace.json` contains non-empty `traceEvents`, including framework spans
   and either a blocking `deepseek_v4_decode_mtp_fused` span or paired asynchronous
   `.worker_submit` / `.worker_wait` spans for every proposed decode step.
7. `server.log` contains host `[STRACE]` records and no `clk=dev` records.
8. `simpler-swimlane.json` and `serving-strace-swimlane.json` contain non-empty
   `traceEvents`.
9. `serving-strace-swimlane.json` contains the serving/framework categories plus
   `strace.host` spans on exactly eight device tracks.
10. `profile-summary.json` reports `simpler.device_effective_available: false`.

Report child teardown warnings separately if the HTTP request completed successfully.

## Read the artifacts

- `completion.txt`: generated text
- `completion-response.json`: complete HTTP response
- `run.log`: top-level workflow output
- `server.log`: combined server and Simpler output
- `serving-trace/trace.json`: merged serving/framework Perfetto trace written by
  `POST /stop_profile`
- `simpler-swimlane.json`: Simpler's native per-invocation host trace
- `serving-strace-swimlane.json`: serving/framework tracks and eight detailed host
  STRACE rows aligned on one timeline
- `profile-summary.{json,md}`: serving-span, request, and per-rank host statistics

## Reprocess an existing run

Reprocess logs without occupying NPUs:

```bash
python .agents/skills/profile-dsv4-serving-strace/scripts/analyze_profile.py \
  <artifact-dir> --run-id <optional-id> --expected-tokens <requested-token-count>

python .agents/skills/profile-dsv4-serving-strace/scripts/render_8lane.py \
  <artifact-dir>/simpler-swimlane.json <artifact-dir>/server.log \
  <artifact-dir>/serving-strace-swimlane.json \
  --serving-trace <artifact-dir>/serving-trace/trace.json
```
