# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Compiled-program cache keys for DeepSeek V4."""

from __future__ import annotations

import functools
import hashlib
import inspect
from pathlib import Path

from pypto_serving.model.common.kernel_cache import (
    UNKNOWN,
    KernelCache as KernelCache,
    pypto_version,
    source_fingerprint,
)


@functools.lru_cache(maxsize=None)
def compute_code_fingerprint(pypto_root: str | None) -> str:
    """Fingerprint the DeepSeek kernel sources and serving compile wrapper."""
    try:
        from pypto_serving.model.deepseek.npu_executor import (  # noqa: PLC0415
            _find_pypto_lib_deepseek_v4_dir,
        )

        kernel_dir = _find_pypto_lib_deepseek_v4_dir(pypto_root)
        sources = [
            (path.relative_to(kernel_dir).as_posix(), path)
            for path in kernel_dir.rglob("*.py")
        ]
        executor_source = Path(__file__).resolve().parent / "npu_executor.py"
        sources.append(("serving/npu_executor.py", executor_source))
        return source_fingerprint(sources)
    except Exception:  # noqa: BLE001 - fingerprint failure safely disables reuse
        return f"{pypto_version()}+{UNKNOWN}"


def compute_params_fingerprint(
    name: str,
    jit_fn: object,
    *,
    platform: str,
    block_dim: int | None,
    prefill_seq: int,
    decode_batch: int,
    decode_seq: int,
    decode_tokens: int,
) -> str:
    """Fingerprint the JIT signature and deployment layout."""
    function = getattr(jit_fn, "_func", None)
    if function is None:
        return UNKNOWN
    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return UNKNOWN

    digest = hashlib.sha256()
    digest.update(repr((name, platform, block_dim)).encode())
    digest.update(
        repr(
            (
                "layout",
                int(prefill_seq),
                int(decode_batch),
                int(decode_seq),
                int(decode_tokens),
            )
        ).encode()
    )
    for parameter in signature.parameters.values():
        descriptor = (
            parameter.name,
            parameter.kind.name,
            repr(parameter.annotation),
            repr(parameter.default),
        )
        digest.update(repr(descriptor).encode())
    digest.update(repr(("return", signature.return_annotation)).encode())
    return digest.hexdigest()[:16]
