# Profiling

The `pypto_serving.tools.profile` module records serving and generation activity in the
[Chrome Trace Event Format](https://chromium.googlesource.com/catapult/+/HEAD/tracing/README.md).
It is disabled by default and has low overhead when disabled. When enabled, it
records spans from the HTTP API, scheduler, engine, worker, executor, and NPU
kernel dispatch paths.

Each process writes to a separate JSON Lines fragment. This avoids
cross-process writes to one file and preserves the events that were flushed if
a run exits before the final merge. On a normal shutdown, the entry points
merge the fragments into a single `trace.json` file that can be opened in a
trace viewer such as [Perfetto](https://ui.perfetto.dev/).

## Configuration

Profiling is enabled when either of these environment variables is present:

| Variable | Description |
| --- | --- |
| `SA_PROFILE_OUTPUT` | Output directory or a path ending in `.json`. Defaults to `./profile_out` when only `SA_PROFILE_LEVEL` is set. |
| `SA_PROFILE_LEVEL` | Comma-separated event levels. Supported values are `e2e`, `kernel`, and `verbose`. Defaults to `e2e,kernel`. |

The event levels are:

- `e2e`: request, scheduler, engine, executor, and worker spans.
- `kernel`: NPU kernel dispatch spans.
- `verbose`: enables all levels, including any fine-grained events marked as
  verbose.

For a directory output, the module creates:

```text
/tmp/pypto-profile/
├── fragments/
│   ├── trace.<pid-1>.jsonl
│   └── trace.<pid-2>.jsonl
└── trace.json
```

When `SA_PROFILE_OUTPUT` ends in `.json`, that path is used for the merged
trace and the fragments are stored in a sibling directory. For example,
`SA_PROFILE_OUTPUT=/tmp/run.json` produces `/tmp/run.json` and
`/tmp/run.json.fragments/`.

Use a different output path for each run. Starting a new main process removes
stale `trace.*.jsonl` files from its fragments directory, and merging replaces
the existing trace file. Absolute output paths are recommended, especially
when using the manual merge script.

## Profile Offline Generation

Set the profiling environment variables before running the generation entry
point:

```bash
SA_PROFILE_OUTPUT=/tmp/pypto-profile-offline \
SA_PROFILE_LEVEL=e2e,kernel \
python examples/model/qwen3_14b/npu_generate.py \
  --model-dir /path/to/Qwen3-14B \
  --prompt 'Huawei is' \
  --platform a2a3 \
  --device-id 0 \
  --max-seq-len 512 \
  --max-new-tokens 5 \
  --profile
```

The entry point merges the fragments in its `finally` block, so the completed
trace is normally available at `/tmp/pypto-profile-offline/trace.json` even if
generation raises an exception.

The `--profile` and `--profile-verbose` options are separate from the
`pypto_serving.tools.profile` module configuration. They enable the timing summary printed
by `npu_generate.py`; they do not enable trace collection by themselves.
Conversely, setting an `SA_PROFILE_*` variable enables trace collection even
without either CLI option.

## Profile HTTP Serving

Start the server with profiling enabled:

```bash
SA_PROFILE_OUTPUT=/tmp/pypto-profile-serving \
SA_PROFILE_LEVEL=e2e,kernel \
pypto-serving \
  --model /path/to/Qwen3-14B \
  --backend npu \
  --platform a2a3 \
  --device 0 \
  --max-model-len 512 \
  --port 8899
```

Send the workload to profile, then stop the server gracefully. The API process
waits for its worker processes to stop and merges all process fragments into
`/tmp/pypto-profile-serving/trace.json` during application shutdown.

## Merge Profile Fragments Manually

Use `scripts/merge_profile.sh` when automatic merging did not run, for example
after an interrupted serving process. Pass the same output path that was used
for `SA_PROFILE_OUTPUT`:

```bash
./scripts/merge_profile.sh /tmp/pypto-profile-serving
```

Alternatively, provide the path through the environment:

```bash
SA_PROFILE_OUTPUT=/tmp/pypto-profile-serving \
  ./scripts/merge_profile.sh
```

Stop all profiled processes before running the script so their buffered events
are flushed to the fragments.

The script accepts both directory and `.json` output forms. It locates every
`trace.<pid>.jsonl` fragment, ignores incomplete or malformed lines, and
atomically replaces the merged trace. A successful run prints the event and
fragment counts, for example:

```text
Merged 1136 events from 2 fragments into /tmp/pypto-profile-serving/trace.json
```

The fragments are retained after merging, so the script can be run again. It
fails without changing the trace if the fragments directory contains no trace
fragments.

## Add Instrumentation

The public helpers are available from `pypto_serving.tools.profile`:

```python
from pypto_serving.tools.profile import profile_duration, profile_instant, profile_span


with profile_span(
    "scheduler.schedule",
    cat="scheduler",
    args={"batch_size": batch_size},
):
    schedule_batch()

profile_instant(
    "request.queued",
    cat="request",
    args={"request_id": request_id},
)

profile_duration(
    "kernel.execute",
    dur_us=kernel_time_us,
    cat="kernel",
    level="kernel",
)
```

- `profile_span()` is a context manager that records a complete duration event.
- `profile_instant()` records a point-in-time event.
- `profile_duration()` records an already measured duration in microseconds.
  Pass `ts_us` to set its start timestamp; otherwise the interval ends when the
  helper is called.
- `is_enabled(level)` can guard expensive argument construction or optional
  instrumentation.
- `get_profiler(process_name=...)` initializes the process-local recorder and
  sets the process name shown in the trace viewer.
- `merge_profile()` closes the current process recorder and builds the final
  trace. Call it only after child processes have stopped and flushed their
  fragments.

The `cat` value groups events in the trace viewer, while `level` controls
whether the event is collected. The helpers are no-ops when profiling or their
level is disabled. Keep event arguments small and JSON-serializable to limit
trace size and recording overhead.
