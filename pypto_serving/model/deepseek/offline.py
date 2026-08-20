# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Offline engine configuration for DeepSeek V4 Flash W8A8."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path

from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.config.types import RuntimeConfig
from pypto_serving.model.deepseek.npu_runner import (
    DeepSeekV4CacheLayout,
    build_deepseek_v4_cache_group_specs,
)
from pypto_serving.serving.engine.async_engine import EngineConfig
from pypto_serving.tools.profile import ProfileConfig, create_profile_config


def build_deepseek_v4_offline_engine_config(
    model_dir: str | Path,
    *,
    device_ids: Sequence[int],
    model_id: str | None = None,
    platform: str = "a2a3",
    max_seq_len: int = 512,
    max_new_tokens: int = 32,
    max_num_seqs: int = 32,
    max_num_batched_tokens: int = 512,
    long_prefill_token_threshold: int = 128,
    npu_memory_utilization: float = 0.90,
    weight_dtype: str = "bfloat16",
    kv_dtype: str = "bfloat16",
    enable_mtp: bool = False,
    enable_prefix_cache: bool = True,
    enable_chunked_prefill: bool = True,
    kernel_cache_dir: str | Path | None = None,
    save_kernels_dir: str | Path | None = None,
    pypto_root: str | Path | None = None,
    profile_config: ProfileConfig | None = None,
) -> EngineConfig:
    """Build the single-worker, eight-rank DeepSeek V4 offline topology.

    DeepSeek's attention-DP ranks share one model worker with the EP ranks, so
    offline inference must not model them as eight independent replicas. The
    returned config uses the same overlapped placement and grouped KV-cache
    contract as HTTP serving.
    """
    model_path = Path(model_dir).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json in {model_path}")
    try:
        config_data = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc
    if not isinstance(config_data, dict):
        raise ValueError(f"DeepSeek V4 config must be a JSON object: {config_path}")

    model_type = str(config_data.get("model_type") or "").lower()
    architectures = {
        str(architecture).lower()
        for architecture in (config_data.get("architectures") or [])
    }
    if model_type != "deepseek_v4" and "deepseekv4forcausallm" not in architectures:
        raise ValueError(f"{model_path} is not a DeepSeek V4 checkpoint")
    quantization = config_data.get("quantization_config") or {}
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "compressed-tensors":
        raise ValueError(
            "DeepSeek V4 offline inference requires the W8A8 compressed-tensors checkpoint"
        )

    devices = tuple(int(device_id) for device_id in device_ids)
    layout = DeepSeekV4CacheLayout(
        decode_batch=4 if enable_mtp else 8,
        decode_seq=2 if enable_mtp else 1,
        decode_tokens=8,
    )
    if len(devices) != layout.ranks:
        raise ValueError(f"DeepSeek V4 offline inference requires exactly {layout.ranks} devices")
    if len(set(devices)) != len(devices):
        raise ValueError(f"device_ids must not contain duplicates: {devices}")
    if max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")
    max_supported_seq_len = layout.prefill_csa_state_max_blocks * layout.c4_state_block_size
    if max_seq_len > max_supported_seq_len:
        raise ValueError(
            f"DeepSeek V4 kernels support max_seq_len <= {max_supported_seq_len}, "
            f"got {max_seq_len}"
        )
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    max_global_batch = layout.ranks * layout.decode_batch
    if max_num_seqs <= 0 or max_num_seqs > max_global_batch:
        raise ValueError(
            f"max_num_seqs must be in [1, {max_global_batch}] for "
            f"{'MTP' if enable_mtp else 'autoregressive'} decode"
        )
    if max_num_batched_tokens <= 0:
        raise ValueError("max_num_batched_tokens must be positive")
    if long_prefill_token_threshold <= 0:
        raise ValueError("long_prefill_token_threshold must be positive")
    if not 0.0 < npu_memory_utilization <= 1.0:
        raise ValueError("npu_memory_utilization must be in (0, 1]")

    compress_ratios = config_data.get("compress_ratios")
    num_hidden_layers = int(config_data.get("num_hidden_layers", 43))
    if not isinstance(compress_ratios, list) or len(compress_ratios) != num_hidden_layers + 1:
        raise ValueError(
            "DeepSeek V4 config compress_ratios must include hidden layers plus the MTP/final entry"
        )
    kv_cache_groups = build_deepseek_v4_cache_group_specs(
        num_hidden_layers,
        compress_ratios,
        decode_batch=layout.decode_batch,
        enable_mtp=enable_mtp,
        max_seq_len=max_seq_len,
    )

    parallel_config = ParallelConfig(
        data_parallel_size=layout.ranks,
        tensor_parallel_size=1,
        expert_parallel_size=layout.ranks,
        enable_expert_parallel=True,
        devices=devices,
        placement_mode="overlapped",
    )
    runtime_config = RuntimeConfig(
        page_size=layout.block_size,
        max_batch_size=max_num_seqs,
        max_seq_len=max_seq_len,
        max_new_tokens=max_new_tokens,
        device="cpu",
        kv_dtype=kv_dtype,
        weight_dtype=weight_dtype,
        npu_memory_utilization=npu_memory_utilization,
        max_num_batched_tokens=max_num_batched_tokens,
        num_speculative_tokens=1 if enable_mtp else 0,
        kv_cache_groups=kv_cache_groups,
    )

    executor_kwargs: dict[str, object] = {
        "compile_kernels": True,
        "enable_mtp": enable_mtp,
    }
    resolved_pypto_root = pypto_root or os.environ.get("PYPTO_ROOT")
    resolved_save_dir = save_kernels_dir or os.environ.get("PYPTO_SAVE_KERNELS_DIR")
    if resolved_pypto_root is not None:
        executor_kwargs["pypto_root"] = str(Path(resolved_pypto_root).expanduser().resolve())
    if resolved_save_dir is not None:
        executor_kwargs["save_kernels_dir"] = str(Path(resolved_save_dir).expanduser().resolve())
    if kernel_cache_dir is not None:
        cache_path = Path(kernel_cache_dir).expanduser().resolve()
        if cache_path.exists() and not cache_path.is_dir():
            raise ValueError(f"kernel_cache_dir exists but is not a directory: {cache_path}")
        executor_kwargs["kernel_cache_dir"] = str(cache_path)

    return EngineConfig(
        model_id=model_id or model_path.name,
        model_dir=str(model_path),
        platform=platform,
        device_id=devices[0],
        device_ids=devices,
        parallel_config=parallel_config,
        executor_cls="PyptoDeepSeekV4Executor",
        executor_kwargs=executor_kwargs,
        runtime_config=runtime_config,
        profile_config=profile_config or create_profile_config(enabled=False),
        max_num_running_reqs=max_num_seqs,
        max_num_scheduled_tokens=max_num_batched_tokens,
        long_prefill_token_threshold=long_prefill_token_threshold,
        enable_prefix_cache=enable_prefix_cache,
        enable_chunk_prefill=enable_chunked_prefill,
    )
