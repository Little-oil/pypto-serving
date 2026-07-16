# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def merge_fragments(fragments_dir: Path, trace_file: Path) -> int:
    """Merge per-process JSONL trace fragments into one Chrome trace file."""
    events: list[dict[str, Any]] = []
    if fragments_dir.is_dir():
        for fragment in sorted(fragments_dir.glob("trace.*.jsonl")):
            with fragment.open("r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    try:
                        events.append(json.loads(stripped))
                    except json.JSONDecodeError:
                        continue

    trace_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = trace_file.with_suffix(trace_file.suffix + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f, separators=(",", ":"))
        f.write("\n")
    tmp_file.replace(trace_file)
    return len(events)
