# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Launch pypto-serving with an explicit KV page count."""

from __future__ import annotations

import dataclasses
import importlib
import os


def main() -> int:
    total_kv_pages = int(os.environ.get("PYPTO_DSV4_TOTAL_KV_PAGES", "32"))
    if total_kv_pages <= 0:
        raise ValueError("PYPTO_DSV4_TOTAL_KV_PAGES must be positive")

    cli_main = importlib.import_module("pypto_serving.cli.main")
    # There is no supported CLI or environment override, so the skill intentionally
    # patches this private builder to keep profiling KV allocation deterministic.
    original_build_runtime_config = cli_main._build_runtime_config

    def build_runtime_config(*args, **kwargs):
        runtime = original_build_runtime_config(*args, **kwargs)
        return dataclasses.replace(
            runtime,
            total_kv_pages=total_kv_pages,
        )

    cli_main._build_runtime_config = build_runtime_config
    return cli_main.main()


if __name__ == "__main__":
    raise SystemExit(main())
