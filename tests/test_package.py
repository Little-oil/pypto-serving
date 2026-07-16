# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Package API and kernel-discovery tests."""

from pathlib import Path

import pypto_serving
from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.config.types import GenerateConfig, RuntimeConfig
from pypto_serving.model.model_loader import ModelLoader
from pypto_serving.model.qwen import npu_executor
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine, EngineConfig, ReplicaEngineCore
from pypto_serving.serving.engine.engine import LLMEngine


def _qwen_kernel_dir(root: Path) -> Path:
    kernel_dir = root / "models" / "qwen3" / "14b"
    kernel_dir.mkdir(parents=True)
    return kernel_dir


def test_root_package_exports_public_api() -> None:
    assert pypto_serving.AsyncLLMEngine is AsyncLLMEngine
    assert pypto_serving.EngineConfig is EngineConfig
    assert pypto_serving.GenerateConfig is GenerateConfig
    assert pypto_serving.LLMEngine is LLMEngine
    assert pypto_serving.ModelLoader is ModelLoader
    assert pypto_serving.ParallelConfig is ParallelConfig
    assert pypto_serving.ReplicaEngineCore is ReplicaEngineCore
    assert pypto_serving.RuntimeConfig is RuntimeConfig


def test_qwen_kernel_discovery_prefers_explicit_root(tmp_path: Path, monkeypatch) -> None:
    explicit_root = tmp_path / "explicit"
    explicit_kernel_dir = _qwen_kernel_dir(explicit_root)
    _qwen_kernel_dir(tmp_path / "environment")
    monkeypatch.setenv("PYPTO_ROOT", str(tmp_path / "environment"))

    assert npu_executor._find_pypto_lib_qwen14b_dir(str(explicit_root)) == explicit_kernel_dir


def test_qwen_kernel_discovery_uses_environment_root(tmp_path: Path, monkeypatch) -> None:
    pypto_root = tmp_path / "environment"
    kernel_dir = _qwen_kernel_dir(pypto_root)
    monkeypatch.setenv("PYPTO_ROOT", str(pypto_root))

    assert npu_executor._find_pypto_lib_qwen14b_dir() == kernel_dir


def test_qwen_kernel_discovery_falls_back_to_editable_checkout(tmp_path: Path, monkeypatch) -> None:
    checkout = tmp_path / "checkout"
    module_path = checkout / "pypto_serving" / "model" / "qwen" / "npu_executor.py"
    module_path.parent.mkdir(parents=True)
    kernel_dir = _qwen_kernel_dir(checkout / "pypto-lib")
    monkeypatch.delenv("PYPTO_ROOT", raising=False)
    monkeypatch.setattr(npu_executor, "__file__", str(module_path))

    assert npu_executor._find_pypto_lib_qwen14b_dir() == kernel_dir
