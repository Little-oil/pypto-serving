#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../../.." && pwd)

usage() {
  cat <<'EOF'
Usage: run_profile.sh --model-dir DIR [options]

Options:
  --artifact-dir DIR       Output directory (default: artifacts/<timestamp>)
  --use-compile-cache      Reuse kernels from the serving worker build directory
  --python PATH            Python interpreter (default: current python3/python)
  --devices IDS            Eight comma-separated device IDs (default: allocation env)
  --max-tokens N           Output-token limit (default: 20)
  --prompt TEXT            Completion prompt (default: "Huawei is")
  --served-model-name NAME API model name (default: dsv4-flash-w8a8)
  --run-id ID              Optional scheduler/task identifier recorded in the report
  -h, --help               Show this help

Run this command inside an environment that already owns eight NPUs. Resource
allocation is intentionally outside this script so it works with local shells,
containers, Slurm, Kubernetes, task-submit, and other site-specific launchers.
EOF
}

PYTHON_BIN=${PYPTO_PROFILE_PYTHON:-}
MODEL_DIR=${PYPTO_DSV4_MODEL_DIR:-}
DEVICES=${PYPTO_PROFILE_DEVICES:-${TASK_DEVICE:-${ASCEND_RT_VISIBLE_DEVICES:-}}}
MAX_TOKENS=20
PROMPT="Huawei is"
SERVED_MODEL_NAME=dsv4-flash-w8a8
RUN_ID=${PYPTO_PROFILE_RUN_ID:-unknown}
ARTIFACT_DIR=
USE_COMPILE_CACHE=${PYPTO_USE_COMPILE_CACHE:-0}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifact-dir)
      ARTIFACT_DIR=$2
      shift 2
      ;;
    --use-compile-cache)
      USE_COMPILE_CACHE=1
      shift
      ;;
    --python)
      PYTHON_BIN=$2
      shift 2
      ;;
    --model-dir)
      MODEL_DIR=$2
      shift 2
      ;;
    --devices)
      DEVICES=$2
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS=$2
      shift 2
      ;;
    --prompt)
      PROMPT=$2
      shift 2
      ;;
    --served-model-name)
      SERVED_MODEL_NAME=$2
      shift 2
      ;;
    --run-id)
      RUN_ID=$2
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$MODEL_DIR" ]]; then
  echo "error: --model-dir or PYPTO_DSV4_MODEL_DIR is required" >&2
  exit 2
fi
if [[ -z "$DEVICES" ]]; then
  echo "error: --devices or an allocation device environment variable is required" >&2
  exit 2
fi
if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN=$(command -v python3 || command -v python)
fi
if [[ -z "$ARTIFACT_DIR" ]]; then
  ARTIFACT_DIR="$REPO_ROOT/artifacts/dsv4-serving-profile-$(date +%Y%m%d-%H%M%S)"
fi

ARTIFACT_DIR=$(realpath -m "$ARTIFACT_DIR")
MODEL_DIR=$(realpath -m "$MODEL_DIR")
case "${USE_COMPILE_CACHE,,}" in
  1|true|yes|on)
    USE_COMPILE_CACHE=1
    ;;
  0|false|no|off|"")
    USE_COMPILE_CACHE=0
    ;;
  *)
    echo "error: PYPTO_USE_COMPILE_CACHE must be a boolean value" >&2
    exit 2
    ;;
esac
trap 'printf "Artifacts: %s\n" "$ARTIFACT_DIR"' EXIT
mkdir -p "$ARTIFACT_DIR"

# Host STRACE remains enabled through the runtime log level. Device-domain
# STRACE is disabled to avoid its profiling/readback path. The Python runner
# configures on-demand serving profiling through CLI options and HTTP control.
export PYPTO_RUNTIME_LOG=${PYPTO_RUNTIME_LOG:-v9}
export SIMPLER_DEVICE_STRACE_ENABLE=0
export PTO2_RING_DEP_POOL=${PTO2_RING_DEP_POOL:-131072}
export PTO2_RING_TASK_WINDOW=${PTO2_RING_TASK_WINDOW:-131072}
export PTO2_RING_HEAP=${PTO2_RING_HEAP:-2147483648}
export SIMPLER_OP_EXECUTE_TIMEOUT_US=${SIMPLER_OP_EXECUTE_TIMEOUT_US:-400000000}
export SIMPLER_STREAM_SYNC_TIMEOUT_MS=${SIMPLER_STREAM_SYNC_TIMEOUT_MS:-440000}
export SIMPLER_SCHEDULER_TIMEOUT_MS=${SIMPLER_SCHEDULER_TIMEOUT_MS:-320000}
export SERVING_WORKER_STEP_TIMEOUT=${SERVING_WORKER_STEP_TIMEOUT:-1800}
export NO_PROXY=${NO_PROXY:-127.0.0.1,localhost}
export no_proxy=${no_proxy:-127.0.0.1,localhost}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO_ROOT"
PROFILE_ARGS=(
  --artifact-dir "$ARTIFACT_DIR"
  --model-dir "$MODEL_DIR"
  --devices "$DEVICES"
  --served-model-name "$SERVED_MODEL_NAME"
  --max-tokens "$MAX_TOKENS"
  --prompt "$PROMPT"
)
if [[ "$USE_COMPILE_CACHE" == 1 ]]; then
  PROFILE_ARGS+=(--use-compile-cache)
fi
"$PYTHON_BIN" "$SCRIPT_DIR/run_profile.py" \
  "${PROFILE_ARGS[@]}" \
  2>&1 | tee "$ARTIFACT_DIR/run.log"

"$PYTHON_BIN" "$SCRIPT_DIR/analyze_profile.py" "$ARTIFACT_DIR" \
  --run-id "$RUN_ID" --expected-tokens "$MAX_TOKENS"
"$PYTHON_BIN" "$SCRIPT_DIR/render_8lane.py" \
  "$ARTIFACT_DIR/simpler-swimlane.json" "$ARTIFACT_DIR/server.log" \
  "$ARTIFACT_DIR/serving-strace-swimlane.json" \
  --serving-trace "$ARTIFACT_DIR/serving-trace/trace.json"
