# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the compiled-kernel disk cache.

These exercise the cache key / marker logic without an NPU or the real pypto
compiler: fake tensors stand in for the compile-time dummy args, and a fake
``DistributedCompiledProgram`` stands in for the reload path.
"""

import types

from pypto_serving.model.deepseek.kernel_cache import (
    compute_params_fingerprint as compute_deepseek_params_fingerprint,
)

def test_deepseek_params_fingerprint_tracks_deployment_layout():
    def kernel(x: tuple[int, int]):
        pass

    jit_fn = types.SimpleNamespace(_func=kernel)
    kwargs = {
        "platform": "a2a3",
        "block_dim": None,
        "prefill_seq": 128,
        "decode_batch": 4,
        "decode_seq": 2,
        "decode_tokens": 8,
    }
    base = compute_deepseek_params_fingerprint("deepseek_v4_decode", jit_fn, **kwargs)

    for dimension in ("prefill_seq", "decode_batch", "decode_seq", "decode_tokens"):
        changed = {**kwargs, dimension: kwargs[dimension] + 1}
        assert base != compute_deepseek_params_fingerprint(
            "deepseek_v4_decode",
            jit_fn,
            **changed,
        )


def test_deepseek_params_fingerprint_tracks_signature_annotations():
    def small(x: tuple[int, int]):
        pass

    def large(x: tuple[int, int, int]):
        pass

    kwargs = {
        "platform": "a2a3",
        "block_dim": None,
        "prefill_seq": 128,
        "decode_batch": 4,
        "decode_seq": 2,
        "decode_tokens": 8,
    }
    small_fingerprint = compute_deepseek_params_fingerprint(
        "deepseek_v4_decode",
        types.SimpleNamespace(_func=small),
        **kwargs,
    )
    large_fingerprint = compute_deepseek_params_fingerprint(
        "deepseek_v4_decode",
        types.SimpleNamespace(_func=large),
        **kwargs,
    )
    assert small_fingerprint != large_fingerprint
