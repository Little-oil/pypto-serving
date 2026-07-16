#!/usr/bin/env bash
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

profile_output="${1:-${SA_PROFILE_OUTPUT:-}}"
if [[ -z "$profile_output" ]]; then
    echo "Usage: $0 PROFILE_OUTPUT" >&2
    echo "       SA_PROFILE_OUTPUT=PROFILE_OUTPUT $0" >&2
    exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python - "$profile_output" <<'PY'
import sys

from pypto_serving.tools.profile.env import load_profile_config
from pypto_serving.tools.profile.merge import merge_fragments


config = load_profile_config({"SA_PROFILE_OUTPUT": sys.argv[1]})
fragments = sorted(config.fragments_dir.glob("trace.*.jsonl"))
if not fragments:
    raise SystemExit(f"No trace fragments found under {config.fragments_dir}")

event_count = merge_fragments(config.fragments_dir, config.trace_file)
print(f"Merged {event_count} events from {len(fragments)} fragments into {config.trace_file}")
PY
