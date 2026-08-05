# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from pathlib import Path

import torch

from pypto_serving.model.qwen.kernel_cache import compute_params_fingerprint
from pypto_serving.model.qwen.npu_executor import Qwen314BPyptoExecutor as PyptoExecutor


ROOT = Path(__file__).resolve().parents[1]
QWEN3_DISPATCH = ROOT / "pypto_serving" / "model" / "qwen" / "qwen3_l3_dispatch.py"
QWEN3_KERNEL_DIR = ROOT / "pypto-lib" / "models" / "qwen3" / "14b"


def test_qwen_compile_uses_current_distributed_config_interface(monkeypatch):
    import pypto.ir.distributed_compiled_program as distributed_program

    class StrictDistributedConfig:
        """Mirror runtimes that retain AICPU tuning but remove block_dim."""

        def __init__(self, *, device_ids, num_sub_workers, aicpu_thread_num):
            self.device_ids = device_ids
            self.num_sub_workers = num_sub_workers
            self.aicpu_thread_num = aicpu_thread_num

    captured = {}

    class FakeJitFunction:
        def compile(self, *args, config):
            captured["config"] = config
            return distributed_program.DistributedCompiledProgram.__new__(
                distributed_program.DistributedCompiledProgram
            )

    monkeypatch.setattr(distributed_program, "DistributedConfig", StrictDistributedConfig)
    executor = PyptoExecutor(device_ids=(3,))

    callable_spec = executor._compile_jit_fwd_callable(
        "fake",
        FakeJitFunction(),
        [],
    )

    run_config = captured["config"]
    assert not hasattr(run_config, "block_dim")
    assert run_config.distributed_config.device_ids == [3]
    assert run_config.distributed_config.num_sub_workers == 0
    assert run_config.distributed_config.aicpu_thread_num == 4
    assert callable_spec.aicpu_thread_num == 4


def test_qwen_kernel_cache_hit_uses_current_callable_interface():
    cached_program = object()
    captured = {}

    class FakeKernelCache:
        def load(self, name, params_fingerprint, *, platform, distributed_config):
            captured["name"] = name
            captured["params_fingerprint"] = params_fingerprint
            captured["platform"] = platform
            captured["distributed_config"] = distributed_config
            return cached_program

    class CompileMustNotRun:
        def compile(self, *args, config):
            raise AssertionError("a kernel-cache hit must skip JIT compilation")

    executor = PyptoExecutor(device_ids=(3,))
    executor._kernel_cache = FakeKernelCache()

    callable_spec = executor._compile_jit_fwd_callable(
        "fake",
        CompileMustNotRun(),
        [torch.empty((2, 4), dtype=torch.bfloat16)],
    )

    assert callable_spec.compiled is cached_program
    assert callable_spec.name == "fake"
    assert callable_spec.aicpu_thread_num == 4
    assert not hasattr(callable_spec, "block_dim")
    assert captured["name"] == "fake"
    assert captured["params_fingerprint"] == compute_params_fingerprint(
        "fake",
        [torch.empty((2, 4), dtype=torch.bfloat16)],
        platform=executor._platform,
    )
    assert captured["platform"] == executor._platform
    assert captured["distributed_config"].device_ids == [3]
