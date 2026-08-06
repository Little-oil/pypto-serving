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

import pytest

from pypto_serving.model.deepseek.offline import build_deepseek_v4_offline_engine_config


def _write_config(model_dir, *, quant_method: str = "compressed-tensors") -> None:
    config = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "num_hidden_layers": 43,
        "compress_ratios": [0, 0, *(4 if index % 2 == 0 else 128 for index in range(2, 43)), 0],
        "quantization_config": {"quant_method": quant_method},
    }
    (model_dir / "config.json").write_text(json.dumps(config))


def test_offline_config_uses_one_overlapped_eight_rank_worker(tmp_path):
    model_dir = tmp_path / "dsv4"
    model_dir.mkdir()
    _write_config(model_dir)

    config = build_deepseek_v4_offline_engine_config(
        model_dir,
        device_ids=tuple(range(8)),
        max_num_seqs=64,
        kernel_cache_dir=tmp_path / "kernel-cache",
    )

    assert config.executor_cls == "PyptoDeepSeekV4Executor"
    assert config.parallel_config.placement_mode == "overlapped"
    assert config.parallel_config.replica_device_groups == (tuple(range(8)),)
    assert config.worker_device_ids() == tuple(range(8))
    assert config.runtime_config.max_batch_size == 64
    assert config.runtime_config.num_speculative_tokens == 0
    assert len(config.runtime_config.kv_cache_groups) == 6
    assert all(group.num_partitions == 8 for group in config.runtime_config.kv_cache_groups)
    assert config.enable_prefix_cache is False
    assert config.executor_kwargs == {
        "compile_kernels": True,
        "enable_mtp": False,
        "kernel_cache_dir": str((tmp_path / "kernel-cache").resolve()),
    }


def test_offline_mtp_config_uses_b4s2_capacity(tmp_path):
    model_dir = tmp_path / "dsv4"
    model_dir.mkdir()
    _write_config(model_dir)

    config = build_deepseek_v4_offline_engine_config(
        model_dir,
        device_ids=tuple(range(8)),
        enable_mtp=True,
        max_num_seqs=32,
    )

    assert config.runtime_config.num_speculative_tokens == 1
    assert config.executor_kwargs["enable_mtp"] is True
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        build_deepseek_v4_offline_engine_config(
            model_dir,
            device_ids=tuple(range(8)),
            enable_mtp=True,
            max_num_seqs=33,
        )


def test_offline_config_rejects_wrong_checkpoint_and_device_count(tmp_path):
    model_dir = tmp_path / "dsv4"
    model_dir.mkdir()
    _write_config(model_dir, quant_method="none")

    with pytest.raises(ValueError, match="W8A8"):
        build_deepseek_v4_offline_engine_config(model_dir, device_ids=tuple(range(8)))

    _write_config(model_dir)
    with pytest.raises(ValueError, match="exactly 8 devices"):
        build_deepseek_v4_offline_engine_config(model_dir, device_ids=(0,))
