# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import math
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor

from pypto_serving.config.types import (
    DecodeBatch,
    DecodeResult,
    ModelConfig,
    PrefillBatch,
    PrefillResult,
    RuntimeConfig,
    RuntimeModel,
)
from pypto_serving.model.common.runner.model_runner import ModelRunner
from pypto_serving.model.deepseek.weight_loader import (
    DeepSeekV4GlobalWeights,
    DeepSeekV4MtpWeights,
    DeepSeekV4StackedLayerWeights,
    DeepSeekV4WeightStore,
)


logger = logging.getLogger(__name__)


DEEPSEEK_V4_RANKS = 8
DEEPSEEK_V4_HC_MULT = 4
DEEPSEEK_V4_BLOCK_SIZE = 128
DEEPSEEK_V4_DECODE_BATCH = 4
DEEPSEEK_V4_DECODE_SEQ = 2
DEEPSEEK_V4_DECODE_TOKENS = DEEPSEEK_V4_DECODE_BATCH * DEEPSEEK_V4_DECODE_SEQ
DEEPSEEK_V4_PREFILL_BATCH = 1
DEEPSEEK_V4_PREFILL_SEQ = 128
# Prefill and decode share the same fixed global physical pools. Request
# ownership is represented only by the block tables; it never expands a cache
# tensor by decode_batch/cache slots.
DEEPSEEK_V4_PREFILL_ORI_MAX_BLOCKS = 128
DEEPSEEK_V4_DECODE_ORI_MAX_BLOCKS = 128
DEEPSEEK_V4_ORI_TABLE_MAX_BLOCKS = 128
DEEPSEEK_V4_SLIDING_WINDOW = 128
DEEPSEEK_V4_CMP_MAX_BLOCKS = 32
DEEPSEEK_V4_IDX_MAX_BLOCKS = 64
DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS = 64
DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS = 65
DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS = 65
DEEPSEEK_V4_C128_STATE_BLOCK_SIZE = 8
DEEPSEEK_V4_C4_STATE_BLOCK_SIZE = 4
DEEPSEEK_V4_PREFILL_CMP_MAX_BLOCKS = DEEPSEEK_V4_CMP_MAX_BLOCKS
DEEPSEEK_V4_PREFILL_IDX_MAX_BLOCKS = DEEPSEEK_V4_IDX_MAX_BLOCKS
DEEPSEEK_V4_PREFILL_HCA_STATE_MAX_BLOCKS = 2048
DEEPSEEK_V4_PREFILL_CSA_STATE_MAX_BLOCKS = 4096
DEEPSEEK_V4_PREFILL_CSA_INNER_STATE_MAX_BLOCKS = 4096
DEEPSEEK_V4_HEAD_DIM = 512
DEEPSEEK_V4_IDX_HEAD_DIM = 128
DEEPSEEK_V4_HCA_MAIN_OUT_DIM = 512
DEEPSEEK_V4_CSA_MAIN_OUT_DIM = 1024
DEEPSEEK_V4_CSA_INNER_OUT_DIM = 256
DEEPSEEK_V4_HCA_STATE_DIM = 2 * DEEPSEEK_V4_HCA_MAIN_OUT_DIM
DEEPSEEK_V4_CSA_STATE_DIM = 2 * DEEPSEEK_V4_CSA_MAIN_OUT_DIM
DEEPSEEK_V4_CSA_INNER_STATE_DIM = 2 * DEEPSEEK_V4_CSA_INNER_OUT_DIM
DEEPSEEK_V4_RMS_NORM_EPS = 1e-6
DEEPSEEK_V4_HC_EPS = 1e-6
# Layer-stacking counts for the packed all-layer decode_fwd kernel.
DEEPSEEK_V4_FWD_NUM_LAYERS = 43
DEEPSEEK_V4_CSA_NUM_LAYERS = 21
DEEPSEEK_V4_HCA_NUM_LAYERS = 20

# Policy values indicate whether a resident argument contains mutable request
# cache state and therefore must be invalidated before its Host backing is
# reused for a later request.
_MAIN_STATIC_RESIDENT_POLICY = {
    "freqs_cos": False,
    "freqs_sin": False,
    "hc_head_fn": False,
    "hc_head_scale": False,
    "hc_head_base": False,
    "final_norm_w": False,
}
_MAIN_CACHE_RESIDENT_POLICY = {
    "kv_cache": True,
    "cmp_kv": True,
    "idx_kv_cache": True,
    "idx_kv_scale": True,
    "hca_compress_state": True,
    "csa_compress_state": True,
    "csa_inner_compress_state": True,
}
_PREFILL_RESIDENT_POLICY = {
    **_MAIN_STATIC_RESIDENT_POLICY,
    **_MAIN_CACHE_RESIDENT_POLICY,
}
_DECODE_RESIDENT_POLICY = {
    **_MAIN_STATIC_RESIDENT_POLICY,
    **_MAIN_CACHE_RESIDENT_POLICY,
}
_MTP_RESIDENT_POLICY = {
    "freqs_cos": False,
    "freqs_sin": False,
    "kv_cache": True,
}


# Argument order for the packed all-43-layer ``l3_prefill_fwd`` kernel. This
# mirrors pypto-lib prefill_fwd.py ``l3_prefill_fwd`` host signature: every
# layer-stacked weight/state tensor in core-parameter order, followed by the
# ``hc_head`` collapse weights, final RMSNorm input and an ``x_out`` output. The
# kernel stops before LM-head; logits are computed on the host from selected
# normalized hidden rows. A trailing ``num_tokens`` scalar is appended at dispatch.
# The work caches
# (kv_cache/cmp_kv/idx_kv_cache) are kernel ``pl.Out`` tensors; weights and
# metadata are inputs.
_PREFILL_FWD_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "kv_cache",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "cmp_kv",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_hadamard_idx",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "idx_kv_cache",
    "idx_kv_scale",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "freqs_cos",
    "freqs_sin",
    "ori_block_table",
    "cmp_block_table",
    "idx_block_table",
    "ori_slot_mapping",
    "position_ids",
    "input_ids",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_w",
    "pre_hc_hidden_out",
    "x_out",
)

# Argument order for the packed all-43-layer ``l3_decode_fwd`` kernel. This
# mirrors pypto-lib decode_fwd.py ``l3_decode_fwd`` host signature: after the
# ``hc_head`` collapse weights the kernel performs final RMSNorm and writes
# normalized ``x_out``. LM-head is computed on the host side.
_DECODE_FWD_TENSOR_ORDER = (
    "x_hc",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_attn_base",
    "attn_norm_w",
    "wq_a",
    "wq_b",
    "wq_b_scale",
    "wkv",
    "gamma_cq",
    "gamma_ckv",
    "kv_cache",
    "attn_sink",
    "wo_a",
    "wo_b",
    "wo_b_scale",
    "hca_cmp_wkv",
    "hca_cmp_wgate",
    "hca_cmp_ape",
    "hca_cmp_norm_w",
    "hca_compress_state",
    "csa_cmp_wkv",
    "csa_cmp_wgate",
    "csa_cmp_ape",
    "csa_cmp_norm_w",
    "csa_compress_state",
    "csa_idx_wq_b",
    "csa_idx_wq_b_scale",
    "csa_weights_proj",
    "csa_hadamard_idx",
    "csa_inner_wkv",
    "csa_inner_wgate",
    "csa_inner_ape",
    "csa_inner_norm_w",
    "csa_inner_compress_state",
    "cmp_kv",
    "idx_kv_cache",
    "idx_kv_scale",
    "hc_ffn_fn",
    "hc_ffn_scale",
    "hc_ffn_base",
    "norm_w",
    "gate_w",
    "gate_bias",
    "tid2eid",
    "routed_w1",
    "routed_w1_scale",
    "routed_w3",
    "routed_w3_scale",
    "routed_w2",
    "routed_w2_scale",
    "shared_w1",
    "shared_w1_scale",
    "shared_w3",
    "shared_w3_scale",
    "shared_w2",
    "shared_w2_scale",
    "freqs_cos",
    "freqs_sin",
    "block_table",
    "ori_slot_mapping",
    "window_swa_indices",
    "window_swa_lens",
    "swa_slot_mapping",
    "swa_indices",
    "swa_lens",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
    "position_ids",
    "kv_seq_lens",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "cmp_block_table",
    "idx_block_table",
    "input_ids",
    "hc_head_fn",
    "hc_head_scale",
    "hc_head_base",
    "final_norm_w",
    "pre_hc_hidden_out",
    "x_out",
)

_MTP_PREFILL_TENSOR_ORDER = (
    "hidden_states", "prev_hidden_states",
    "enorm_w", "hnorm_w", "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
    "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "kv_cache", "ori_block_table", "ori_slot_mapping",
    "position_ids", "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale", "shared_w1", "shared_w1_scale",
    "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
    "mtp_hc_head_fn", "mtp_hc_head_scale", "mtp_hc_head_base", "mtp_norm_w",
    "hidden_out", "pre_hc_hidden_out",
)

_MTP_DECODE_TENSOR_ORDER = (
    "hidden_states", "prev_pre_hc_hidden", "position_ids",
    "enorm_w", "hnorm_w", "e_proj_w", "e_proj_w_scale", "e_proj_smooth",
    "h_proj_w", "h_proj_w_scale", "h_proj_smooth",
    "hc_attn_fn", "hc_attn_scale", "hc_attn_base", "attn_norm_w",
    "wq_a", "wq_b", "wq_b_scale", "wkv", "gamma_cq", "gamma_ckv",
    "freqs_cos", "freqs_sin", "kv_cache", "swa_slot_mapping", "swa_indices", "swa_lens",
    "attn_sink", "wo_a", "wo_b", "wo_b_scale",
    "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base", "norm_w",
    "gate_w", "gate_bias", "tid2eid", "input_ids",
    "routed_w1", "routed_w1_scale", "routed_w3", "routed_w3_scale",
    "routed_w2", "routed_w2_scale", "shared_w1", "shared_w1_scale",
    "shared_w3", "shared_w3_scale", "shared_w2", "shared_w2_scale",
    "mtp_hc_head_fn", "mtp_hc_head_scale", "mtp_hc_head_base", "mtp_norm_w",
    "hidden_out", "next_pre_hc_hidden",
)

_DECODE_INPUT_TENSOR_FIELDS = (
    "input_ids",
    "position_ids",
    "kv_seq_lens",
    "block_table",
    "ori_slot_mapping",
    "window_swa_indices",
    "window_swa_lens",
    "swa_slot_mapping",
    "swa_indices",
    "swa_lens",
    "cmp_block_table",
    "idx_block_table",
    "hca_compress_state_block_table",
    "csa_compress_state_block_table",
    "csa_inner_compress_state_block_table",
    "hca_cmp_slot_mapping",
    "hca_state_slot_mapping",
    "csa_cmp_slot_mapping",
    "csa_idx_slot_mapping",
    "csa_state_slot_mapping",
    "csa_inner_state_slot_mapping",
)


@dataclass(frozen=True)
class DeepSeekV4CacheLayout:
    """Static cache layout baked into the current DeepSeekV4 kernels."""

    ranks: int = DEEPSEEK_V4_RANKS
    hc_mult: int = DEEPSEEK_V4_HC_MULT
    block_size: int = DEEPSEEK_V4_BLOCK_SIZE
    decode_batch: int = DEEPSEEK_V4_DECODE_BATCH
    decode_seq: int = DEEPSEEK_V4_DECODE_SEQ
    decode_tokens: int = DEEPSEEK_V4_DECODE_TOKENS
    prefill_batch: int = DEEPSEEK_V4_PREFILL_BATCH
    prefill_seq: int = DEEPSEEK_V4_PREFILL_SEQ
    prefill_ori_max_blocks: int = DEEPSEEK_V4_PREFILL_ORI_MAX_BLOCKS
    decode_ori_max_blocks: int = DEEPSEEK_V4_DECODE_ORI_MAX_BLOCKS
    ori_table_max_blocks: int = DEEPSEEK_V4_ORI_TABLE_MAX_BLOCKS
    sliding_window: int = DEEPSEEK_V4_SLIDING_WINDOW
    cmp_max_blocks: int = DEEPSEEK_V4_CMP_MAX_BLOCKS
    idx_max_blocks: int = DEEPSEEK_V4_IDX_MAX_BLOCKS
    hca_state_max_blocks: int = DEEPSEEK_V4_HCA_STATE_MAX_BLOCKS
    csa_state_max_blocks: int = DEEPSEEK_V4_CSA_STATE_MAX_BLOCKS
    csa_inner_state_max_blocks: int = DEEPSEEK_V4_CSA_INNER_STATE_MAX_BLOCKS
    c128_state_block_size: int = DEEPSEEK_V4_C128_STATE_BLOCK_SIZE
    c4_state_block_size: int = DEEPSEEK_V4_C4_STATE_BLOCK_SIZE
    prefill_cmp_max_blocks: int = DEEPSEEK_V4_PREFILL_CMP_MAX_BLOCKS
    prefill_idx_max_blocks: int = DEEPSEEK_V4_PREFILL_IDX_MAX_BLOCKS
    prefill_hca_state_max_blocks: int = DEEPSEEK_V4_PREFILL_HCA_STATE_MAX_BLOCKS
    prefill_csa_state_max_blocks: int = DEEPSEEK_V4_PREFILL_CSA_STATE_MAX_BLOCKS
    prefill_csa_inner_state_max_blocks: int = DEEPSEEK_V4_PREFILL_CSA_INNER_STATE_MAX_BLOCKS

    @property
    def prefill_cmp_block_num(self) -> int:
        """Physical cmp_kv blocks per layer in the packed prefill kernel."""
        return self.prefill_cmp_max_blocks

    @property
    def prefill_idx_block_num(self) -> int:
        """Physical idx_kv_cache blocks per CSA layer in the packed prefill kernel."""
        return self.prefill_idx_max_blocks

    def validate_runtime(self, config: ModelConfig, runtime: RuntimeConfig, device_ids: Sequence[int]) -> None:
        """Validate serving/runtime options against kernel-fixed dimensions."""
        if len(device_ids) != self.ranks:
            raise ValueError(f"DeepSeekV4 requires exactly {self.ranks} devices, got {len(device_ids)}")
        if runtime.page_size != self.block_size:
            raise ValueError(f"DeepSeekV4 kernels require page_size={self.block_size}, got {runtime.page_size}")
        if runtime.max_batch_size > self.decode_batch:
            raise ValueError(
                f"DeepSeekV4 decode kernels support at most {self.decode_batch} active rows, "
                f"got max_batch_size={runtime.max_batch_size}"
            )
        decode_state_capacity = self.prefill_csa_state_max_blocks * self.c4_state_block_size
        if runtime.max_seq_len > decode_state_capacity:
            raise ValueError(
                "DeepSeekV4 pypto-lib decode CSA state tables currently support at most "
                f"max_seq_len={decode_state_capacity}, got {runtime.max_seq_len}. "
                "Increase the decode CSA state table depth in pypto-lib before serving longer contexts."
            )
        if self.decode_tokens != self.decode_batch * self.decode_seq:
            raise ValueError("DeepSeekV4 layout decode_tokens must equal decode_batch * decode_seq")
        expected = {
            "hidden_size": 4096,
            "num_hidden_layers": 43,
            "num_attention_heads": 64,
            "num_key_value_heads": 1,
            "head_dim": 512,
            "vocab_size": 129280,
        }
        actual = {
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "vocab_size": config.vocab_size,
        }
        if actual != expected:
            mismatch = ", ".join(f"{name}={actual[name]} expected {value}" for name, value in expected.items())
            raise ValueError("DeepSeekV4 W8A8 kernels require Flash shape: " + mismatch)


@dataclass
class DeepSeekV4CacheManager:
    """Request-to-cache-slot mapping and table builders for DeepSeekV4 kernels."""

    layout: DeepSeekV4CacheLayout = field(default_factory=DeepSeekV4CacheLayout)
    _request_to_slot: dict[str, int] = field(default_factory=dict)
    _free_slots: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self._free_slots:
            self._free_slots = list(range(self.layout.decode_batch))

    @property
    def active_slots(self) -> dict[str, int]:
        """Return a copy of currently assigned request slots."""
        return dict(self._request_to_slot)

    @property
    def free_count(self) -> int:
        """Return the number of unassigned decode slots."""
        return len(self._free_slots)

    def allocate(self, request_id: str) -> int | None:
        """Assign a stable decode slot to ``request_id``."""
        if request_id in self._request_to_slot:
            return self._request_to_slot[request_id]
        if not self._free_slots:
            return None
        slot = self._free_slots.pop(0)
        self._request_to_slot[request_id] = slot
        return slot

    def release(self, request_ids: Iterable[str]) -> None:
        """Release slots held by finished or aborted requests."""
        for request_id in request_ids:
            slot = self._request_to_slot.pop(request_id, None)
            if slot is not None and slot not in self._free_slots:
                self._free_slots.append(slot)
        self._free_slots.sort()

    def slots_for_request_ids(self, request_ids: Sequence[str]) -> list[int]:
        """Return assigned slots for request ids, allocating missing slots."""
        slots = []
        for request_id in request_ids:
            slot = self.allocate(request_id)
            if slot is None:
                raise RuntimeError("DeepSeekV4 cache slots exhausted")
            slots.append(slot)
        return slots

    def block_table(
        self,
        slots: Sequence[int],
        *,
        max_blocks: int,
        physical_blocks: int | None = None,
    ) -> torch.Tensor:
        """Build logical-to-physical tables for one fixed global cache pool."""
        physical_blocks = int(max_blocks if physical_blocks is None else physical_blocks)
        if physical_blocks <= 0:
            raise ValueError("physical_blocks must be positive")
        cols = torch.arange(int(max_blocks), dtype=torch.int32) % physical_blocks
        table = torch.empty((len(slots), int(max_blocks)), dtype=torch.int32)
        for row, slot in enumerate(slots):
            table[row].copy_((cols * self.layout.decode_batch + int(slot)) % physical_blocks)
        return table

    def slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        block_size: int | None = None,
        compress_ratio: int = 1,
    ) -> torch.Tensor:
        """Map logical token positions to physical cache rows for each request slot."""
        block_size = self.layout.block_size if block_size is None else int(block_size)
        if compress_ratio <= 0:
            raise ValueError("compress_ratio must be positive")
        capacity = max_blocks * block_size
        table = self.block_table(slots, max_blocks=max_blocks)
        max_tokens = max((len(row) for row in positions), default=0)
        mapping = torch.full((len(slots), max_tokens), -1, dtype=torch.int64)
        for row, (slot, row_positions) in enumerate(zip(slots, positions, strict=True)):
            for col, position in enumerate(row_positions):
                logical = int(position) // compress_ratio
                if logical >= capacity:
                    raise ValueError(
                        f"position {position} maps to logical cache row {logical}, "
                        f"but capacity is {capacity}"
                    )
                logical_block, intra = divmod(logical, block_size)
                mapping[row, col] = int(table[row, logical_block]) * block_size + intra
        return mapping

    def block_table_for_kernel_rows(
        self,
        slots: Sequence[int],
        *,
        max_blocks: int,
        kernel_rows: int,
        physical_blocks: int | None = None,
    ) -> torch.Tensor:
        """Build a fixed-row block table, replicating row 0 into inactive rows."""
        if not slots:
            raise ValueError("slots must not be empty")
        active = self.block_table(
            slots,
            max_blocks=max_blocks,
            physical_blocks=physical_blocks,
        )
        return self.replicate_first_row(active, actual_rows=len(slots), kernel_rows=kernel_rows)

    def sliding_window_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Map absolute positions into the fixed global ori-KV pool."""
        rows = self._replicated_slots_and_positions(slots, positions, kernel_rows=kernel_rows)
        table = self.block_table(
            [slot for slot, _ in rows],
            max_blocks=self.layout.ori_table_max_blocks,
            physical_blocks=self.layout.decode_ori_max_blocks,
        )
        mapping = torch.full((kernel_rows, max((len(row) for _, row in rows), default=0)), -1, dtype=torch.int64)
        for row_idx, (_, row_positions) in enumerate(rows):
            for col, position in enumerate(row_positions):
                logical_block, intra = divmod(int(position), self.layout.block_size)
                mapping[row_idx, col] = int(table[row_idx, logical_block]) * self.layout.block_size + intra
        return mapping

    def paged_ori_block_table(
        self,
        slots: Sequence[int],
        *,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Build the decode ori-KV block table (vLLM-style absolute logical columns).

        Each row owns a small physical sliding-window ring of
        ``decode_ori_max_blocks`` pages at physical base ``slot * ring``; the
        ``ori_table_max_blocks`` absolute logical columns wrap into that ring via
        ``physical = slot * ring + (logical % ring)``. This mirrors pypto-lib's
        ``decode_metadata.block_table`` but keys the physical base off the serving
        cache slot (rows already indirect kernel row -> slot).
        """
        padded = self._padded_decode_slots(slots, kernel_rows=kernel_rows)
        return self.block_table(
            padded,
            max_blocks=self.layout.ori_table_max_blocks,
            physical_blocks=self.layout.decode_ori_max_blocks,
        )

    def _padded_decode_slots(self, slots: Sequence[int], *, kernel_rows: int) -> list[int]:
        padded = [int(slot) for slot in slots]
        if not padded:
            raise ValueError("decode must include at least one slot")
        if len(padded) > kernel_rows:
            raise ValueError("active rows exceed kernel_rows")
        padded.extend(padded[0] for _ in range(kernel_rows - len(padded)))
        return padded

    def paged_decode_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Absolute paged write position for the decode SWA ori-KV ring.

        ``physical = block_table[row, pos // block_size] * block_size + pos % block_size``.
        """
        block_size = int(self.layout.block_size)
        rows = self._replicated_slots_and_positions(slots, positions, kernel_rows=kernel_rows)
        table = self.block_table(
            [slot for slot, _ in rows],
            max_blocks=self.layout.ori_table_max_blocks,
            physical_blocks=self.layout.decode_ori_max_blocks,
        )
        mapping = torch.full((kernel_rows, max((len(row) for _, row in rows), default=0)), -1, dtype=torch.int64)
        for row_idx, (_, row_positions) in enumerate(rows):
            for col, position in enumerate(row_positions):
                position = int(position)
                logical_blk = position // block_size
                phys_blk = int(table[row_idx, logical_blk])
                mapping[row_idx, col] = phys_blk * block_size + (position % block_size)
        return mapping

    def swa_window_indices_and_lens(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        kernel_rows: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Lower each decode SWA window to physical KV-cache row indices.

        Mirrors pypto-lib ``decode_metadata.swa_indices_and_lens``: each visible
        absolute position in ``[max(0, pos - window + 1), pos]`` is translated
        through the same paged ring block table as the write path. Rows are packed
        oldest-to-newest; invalid tail columns are ``-1`` and the returned lens give
        the valid prefix length. HCA/CSA use this same full-window cache-first
        contract: their current-chunk KV rows are written before sparse attention.
        """
        window = int(self.layout.sliding_window)
        block_size = int(self.layout.block_size)
        rows = self._replicated_slots_and_positions(slots, positions, kernel_rows=kernel_rows)
        table = self.block_table(
            [slot for slot, _ in rows],
            max_blocks=self.layout.ori_table_max_blocks,
            physical_blocks=self.layout.decode_ori_max_blocks,
        )
        # One window row per decode token (T = kernel_rows x decode_seq), packed
        # row-major to match ``ori_slot_mapping.reshape(-1)`` token ordering.
        per_row = max((len(row) for _, row in rows), default=0)
        total = kernel_rows * per_row
        indices = torch.full((total, window), -1, dtype=torch.int32)
        lens = torch.zeros((total,), dtype=torch.int32)
        for row_idx, (_, row_positions) in enumerate(rows):
            for s, position in enumerate(row_positions):
                token = row_idx * per_row + s
                abs_pos = int(position)
                start = max(0, abs_pos - window + 1)
                out_k = 0
                for pos in range(start, abs_pos + 1):
                    logical_blk = pos // block_size
                    phys_blk = int(table[row_idx, logical_blk])
                    indices[token, out_k] = phys_blk * block_size + (pos % block_size)
                    out_k += 1
                lens[token] = out_k
        return indices, lens

    def compressed_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        compress_ratio: int,
        kernel_rows: int,
    ) -> torch.Tensor:
        """Map compression-boundary positions into a compressed KV cache."""
        rows = self._replicated_slots_and_positions(slots, positions, kernel_rows=kernel_rows)
        mapping = torch.full((kernel_rows, max((len(row) for _, row in rows), default=0)), -1, dtype=torch.int64)
        capacity = max_blocks * self.layout.block_size
        table = self.block_table(
            [slot for slot, _ in rows],
            max_blocks=max_blocks,
        )
        for row_idx, (_, row_positions) in enumerate(rows):
            for col, position in enumerate(row_positions):
                position = int(position)
                if (position + 1) % compress_ratio != 0:
                    continue
                logical = position // compress_ratio
                if logical >= capacity:
                    raise ValueError(
                        f"position {position} maps to compressed row {logical}, "
                        f"but capacity is {capacity}"
                    )
                logical_block, intra = divmod(logical, self.layout.block_size)
                mapping[row_idx, col] = int(table[row_idx, logical_block]) * self.layout.block_size + intra
        return mapping

    def state_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        max_blocks: int,
        state_block_size: int,
        kernel_rows: int,
        physical_blocks: int | None = None,
    ) -> torch.Tensor:
        """Map absolute token positions into a compressor-state cache."""
        rows = self._replicated_slots_and_positions(slots, positions, kernel_rows=kernel_rows)
        mapping = torch.full((kernel_rows, max((len(row) for _, row in rows), default=0)), -1, dtype=torch.int64)
        capacity = max_blocks * state_block_size
        table = self.block_table(
            [slot for slot, _ in rows],
            max_blocks=max_blocks,
            physical_blocks=physical_blocks,
        )
        for row_idx, (_, row_positions) in enumerate(rows):
            for col, position in enumerate(row_positions):
                position = int(position)
                if position >= capacity:
                    raise ValueError(
                        f"position {position} exceeds compressor-state capacity {capacity} "
                        f"(max_blocks={max_blocks}, state_block_size={state_block_size})"
                    )
                logical_block, intra = divmod(position, state_block_size)
                mapping[row_idx, col] = int(table[row_idx, logical_block]) * state_block_size + intra
        return mapping

    @staticmethod
    def _replicated_slots_and_positions(
        slots: Sequence[int],
        positions: Sequence[Sequence[int]],
        *,
        kernel_rows: int,
    ) -> list[tuple[int, Sequence[int]]]:
        if not slots:
            raise ValueError("slots must not be empty")
        if len(slots) != len(positions):
            raise ValueError("slots and positions must have the same active row count")
        if len(slots) > kernel_rows:
            raise ValueError("active rows exceed kernel_rows")
        rows = [(int(slot), tuple(int(pos) for pos in row)) for slot, row in zip(slots, positions, strict=True)]
        rows.extend((rows[0][0], rows[0][1]) for _ in range(kernel_rows - len(rows)))
        return rows

    @staticmethod
    def replicate_first_row(tensor: torch.Tensor, *, actual_rows: int, kernel_rows: int) -> torch.Tensor:
        """Pad kernel inputs by replicating row 0 into inactive rows."""
        if actual_rows <= 0:
            raise ValueError("actual_rows must be positive")
        if kernel_rows < actual_rows:
            raise ValueError("kernel_rows must be >= actual_rows")
        if tensor.shape[0] < actual_rows:
            raise ValueError("tensor has fewer rows than actual_rows")
        out = torch.empty((kernel_rows, *tensor.shape[1:]), dtype=tensor.dtype)
        out[:actual_rows].copy_(tensor[:actual_rows])
        if actual_rows < kernel_rows:
            out[actual_rows:].copy_(tensor[0:1].expand(kernel_rows - actual_rows, *tensor.shape[1:]))
        return out


class DeepSeekV4InputBuilder:
    """Build fixed-shape host inputs for DeepSeekV4 HC-stack kernels."""

    def __init__(self, *, layout: DeepSeekV4CacheLayout, hidden_size: int) -> None:
        self.layout = layout
        self.hidden_size = int(hidden_size)

    def prefill_x_hc(self, embeddings: torch.Tensor, *, actual_tokens: int) -> torch.Tensor:
        """Build ``[ranks, 128, hc_mult, hidden]`` prefill HC input."""
        if embeddings.ndim != 2:
            raise ValueError(f"prefill embeddings must be rank-2, got shape={tuple(embeddings.shape)}")
        return self._x_hc_from_rows(
            embeddings.to(torch.float32),
            actual_tokens=actual_tokens,
            token_rows=self.layout.prefill_seq,
        )

    def decode_x_hc(
        self,
        embeddings: torch.Tensor,
        *,
        actual_batch: int,
        prev_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Build ``[ranks, 128, hc_mult, hidden]`` decode HC input.

        Current DeepSeekV4 decode kernels use a fixed ``decode_tokens`` contract
        with ``decode_seq`` token slots per request. If ``decode_seq`` is greater
        than one and ``prev_embeddings`` is provided, earlier slots carry the
        previous token and the final slot carries the last token. Padding rows
        still replicate row 0 / their own embedding to keep the fixed rows valid.
        """
        if embeddings.ndim != 2:
            raise ValueError(f"decode embeddings must be rank-2, got shape={tuple(embeddings.shape)}")
        if actual_batch <= 0:
            raise ValueError("actual_batch must be positive")
        if actual_batch > self.layout.decode_batch:
            raise ValueError(
                f"actual_batch={actual_batch} exceeds decode batch capacity {self.layout.decode_batch}"
            )
        if embeddings.shape[0] < actual_batch:
            raise ValueError("decode embeddings has fewer rows than actual_batch")
        if prev_embeddings is not None and prev_embeddings.shape[0] < actual_batch:
            raise ValueError("decode prev_embeddings has fewer rows than actual_batch")
        embeddings = embeddings.to(torch.float32)
        if prev_embeddings is not None:
            prev_embeddings = prev_embeddings.to(torch.float32)
        rows = torch.zeros(
            (self.layout.decode_tokens, self.hidden_size),
            dtype=embeddings.dtype,
            device=embeddings.device,
        )
        decode_seq = self.layout.decode_seq
        # When the caller supplies a full per-row embedding tensor (one row per
        # decode-batch slot), use each row's own embedding so the MoE gate routes
        # the 128 tokens across many experts. Otherwise replicate slot 0 into the
        # padding rows as before.
        per_row = embeddings.shape[0] >= self.layout.decode_batch
        for row in range(self.layout.decode_batch):
            source_row = row if per_row else (row if row < actual_batch else 0)
            start = row * decode_seq
            if prev_embeddings is not None and row < actual_batch:
                # Fill every slot with prev, then overwrite the final slot with
                # the last token.
                rows[start : start + decode_seq].copy_(
                    prev_embeddings[row : row + 1].expand(decode_seq, self.hidden_size)
                )
                rows[start + decode_seq - 1].copy_(embeddings[row])
            else:
                rows[start : start + decode_seq].copy_(
                    embeddings[source_row : source_row + 1].expand(decode_seq, self.hidden_size)
                )
        return self._expand_hc_and_ranks(rows)

    def _x_hc_from_rows(
        self,
        embeddings: torch.Tensor,
        *,
        actual_tokens: int,
        token_rows: int,
    ) -> torch.Tensor:
        if actual_tokens <= 0:
            raise ValueError("actual_tokens must be positive")
        if actual_tokens > token_rows:
            raise ValueError(f"actual_tokens={actual_tokens} exceeds token row capacity {token_rows}")
        if embeddings.shape[0] < actual_tokens:
            raise ValueError("embeddings has fewer rows than actual_tokens")
        if int(embeddings.shape[1]) != self.hidden_size:
            raise ValueError(f"embedding hidden size must be {self.hidden_size}, got {int(embeddings.shape[1])}")
        rows = torch.zeros((token_rows, self.hidden_size), dtype=embeddings.dtype, device=embeddings.device)
        rows[:actual_tokens].copy_(embeddings[:actual_tokens])
        return self._expand_hc_and_ranks(rows)

    def _expand_hc_and_ranks(self, rows: torch.Tensor) -> torch.Tensor:
        return (
            rows.unsqueeze(1)
            .expand(rows.shape[0], self.layout.hc_mult, self.hidden_size)
            .unsqueeze(0)
            .expand(self.layout.ranks, rows.shape[0], self.layout.hc_mult, self.hidden_size)
            .contiguous()
        )


@dataclass
class DeepSeekV4L3Callable:
    """Compiled HOST-dispatched DeepSeekV4 program."""

    compiled: object
    name: str


@dataclass
class _StaticDeviceTensor:
    """Rank-stacked CPU tensor marker uploaded to every chip worker once."""

    tensor: torch.Tensor
    cache_state: bool = False


@dataclass
class _TransientDeviceTensor:
    """CPU tensor marker uploaded for one layer dispatch and then freed."""

    tensor: torch.Tensor


@dataclass
class DeepSeekV4LayerCache:
    """Shared decode work-cache tensors for one DeepSeekV4 layer dispatch."""

    kv_cache: torch.Tensor
    cmp_kv: torch.Tensor
    idx_kv_cache: torch.Tensor
    idx_kv_scale: torch.Tensor
    hca_compress_state: torch.Tensor
    csa_compress_state: torch.Tensor
    csa_inner_compress_state: torch.Tensor


@dataclass
class DeepSeekV4CompiledKernels:
    """Compiled-kernel placeholder and immutable DeepSeekV4 runtime metadata."""

    layout: DeepSeekV4CacheLayout
    model_dir: str
    weight_map: dict[str, str]
    weight_store: DeepSeekV4WeightStore
    compress_ratios: tuple[int, ...]
    layer_plan: tuple["DeepSeekV4LayerPlan", ...]
    kernel_dir: str
    prefill: DeepSeekV4L3Callable | None = None
    decode: DeepSeekV4L3Callable | None = None
    mtp_prefill: DeepSeekV4L3Callable | None = None
    mtp_decode: DeepSeekV4L3Callable | None = None
    freqs_cos: torch.Tensor | None = None
    freqs_sin: torch.Tensor | None = None
    platform: str = "a2a3"
    device_id: int = 0
    n_routed_experts: int = 256
    num_hash_layers: int = 3
    embedding_weight: torch.Tensor | None = None

    def l3_callables(self) -> tuple[DeepSeekV4L3Callable, ...]:
        """Return every compiled L3 program that the shared worker may run."""
        callables: list[DeepSeekV4L3Callable] = []
        if self.prefill is not None:
            callables.append(self.prefill)
        if self.decode is not None:
            callables.append(self.decode)
        if self.mtp_prefill is not None:
            callables.append(self.mtp_prefill)
        if self.mtp_decode is not None:
            callables.append(self.mtp_decode)
        return tuple(callables)


@dataclass(frozen=True)
class DeepSeekV4PreparedPrefillInputs:
    """Fixed-shape host tensors derived from one serving prefill chunk."""

    request_id: str
    slot: int
    actual_tokens: int
    x_hc: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    ori_block_table: torch.Tensor
    ori_slot_mapping: torch.Tensor
    cmp_block_table: torch.Tensor
    idx_block_table: torch.Tensor
    hca_compress_state_block_table: torch.Tensor
    csa_compress_state_block_table: torch.Tensor
    csa_inner_compress_state_block_table: torch.Tensor
    hca_cmp_slot_mapping: torch.Tensor
    hca_state_slot_mapping: torch.Tensor
    csa_cmp_slot_mapping: torch.Tensor
    csa_idx_slot_mapping: torch.Tensor
    csa_state_slot_mapping: torch.Tensor
    csa_inner_state_slot_mapping: torch.Tensor


@dataclass(frozen=True)
class DeepSeekV4PreparedDecodeInputs:
    """Fixed-shape host tensors derived from one decode scheduler batch."""

    request_ids: tuple[str, ...]
    slots: tuple[int, ...]
    kernel_slots: tuple[int, ...]
    actual_batch: int
    x_hc: torch.Tensor
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    kv_seq_lens: torch.Tensor
    block_table: torch.Tensor
    ori_slot_mapping: torch.Tensor
    window_swa_indices: torch.Tensor
    window_swa_lens: torch.Tensor
    swa_slot_mapping: torch.Tensor
    swa_indices: torch.Tensor
    swa_lens: torch.Tensor
    cmp_block_table: torch.Tensor
    idx_block_table: torch.Tensor
    hca_compress_state_block_table: torch.Tensor
    csa_compress_state_block_table: torch.Tensor
    csa_inner_compress_state_block_table: torch.Tensor
    hca_cmp_slot_mapping: torch.Tensor
    hca_state_slot_mapping: torch.Tensor
    csa_cmp_slot_mapping: torch.Tensor
    csa_idx_slot_mapping: torch.Tensor
    csa_state_slot_mapping: torch.Tensor
    csa_inner_state_slot_mapping: torch.Tensor


@dataclass
class _DeepSeekV4DecodeSharedBuffers:
    """Reusable decode shared-memory buffers inherited by the L3 chip workers."""

    x_hc_a: torch.Tensor
    x_hc_b: torch.Tensor
    pre_hc_hidden_out: torch.Tensor
    x_out: torch.Tensor
    tensors: dict[str, torch.Tensor]


@dataclass
class _DeepSeekV4PrefillFwdSharedBuffers:
    """Reusable packed-prefill shared buffers inherited by the L3 chip workers.

    For the single ``l3_prefill_fwd`` dispatch the work caches are flattened 5-D
    (kv_cache/cmp_kv stack across all 43 hidden layers, idx_kv_cache across the 21
    compress_ratio==4 layers) and the compress-state kv/score caches stack across
    the CSA (x21) and HCA (x20) groups. The per-step metadata, RoPE tables and
    compress-state block tables are shared single per-rank copies (the kernel
    slices them per layer). ``tensors`` is keyed by ``_PREFILL_FWD_TENSOR_ORDER``
    name (excluding the stacked weights, which live in ``_stacked_weight_buffers``,
    and ``freqs_*``/``x_hc`` which are tracked explicitly). The final normalized
    hidden output is held separately in ``_prefill_output_buffer``.
    """

    x_hc: torch.Tensor
    freqs_cos: torch.Tensor
    freqs_sin: torch.Tensor
    tensors: dict[str, torch.Tensor]


@dataclass
class _DeepSeekV4MtpSharedBuffers:
    """MTP weights, unified SWA cache, recurrent state, and outputs."""

    weights: dict[str, torch.Tensor]
    prefill_hidden_in: torch.Tensor
    prefill_prev_hidden_in: torch.Tensor
    prefill_input_ids: torch.Tensor
    prefill_position_ids: torch.Tensor
    prefill_block_table: torch.Tensor
    prefill_slot_mapping: torch.Tensor
    prefill_kv_cache: torch.Tensor
    decode_hidden_in: torch.Tensor
    decode_prev_hidden_in: torch.Tensor
    decode_input_ids: torch.Tensor
    decode_position_ids: torch.Tensor
    decode_slot_mapping: torch.Tensor
    decode_swa_indices: torch.Tensor
    decode_swa_lens: torch.Tensor
    decode_kv_cache: torch.Tensor
    prefill_hidden_out: torch.Tensor
    prefill_pre_hc_out: torch.Tensor
    decode_hidden_out: torch.Tensor
    decode_pre_hc_out: torch.Tensor


@dataclass(frozen=True)
class DeepSeekV4LayerPlan:
    """Per-layer execution metadata for DeepSeekV4 serving."""

    layer_id: int
    compress_ratio: int
    attention_kind: str
    include_tid2eid: bool
    include_gate_bias: bool


def deepseek_v4_attention_kind(compress_ratio: int) -> str:
    """Return the DeepSeekV4 attention family for a compression ratio."""
    if compress_ratio == 0:
        return "swa"
    if compress_ratio == 128:
        return "hca"
    if compress_ratio == 4:
        return "csa"
    raise ValueError(f"unsupported DeepSeekV4 attention compress ratio: {compress_ratio}")


def build_deepseek_v4_layer_plan(
    *,
    compress_ratios: Sequence[int],
    num_hidden_layers: int,
    num_hash_layers: int,
) -> tuple[DeepSeekV4LayerPlan, ...]:
    """Build the per-layer serving plan from config metadata."""
    if len(compress_ratios) < num_hidden_layers:
        raise ValueError("compress_ratios must include at least one entry per hidden layer")
    return tuple(
        DeepSeekV4LayerPlan(
            layer_id=layer_id,
            compress_ratio=int(compress_ratios[layer_id]),
            attention_kind=deepseek_v4_attention_kind(int(compress_ratios[layer_id])),
            include_tid2eid=layer_id < num_hash_layers,
            include_gate_bias=layer_id >= num_hash_layers,
        )
        for layer_id in range(num_hidden_layers)
    )


def accept_mtp_tokens(main_token_ids: torch.Tensor, draft_token_ids: torch.Tensor) -> list[list[int]]:
    """Accept one MTP draft against two-token main-model greedy predictions.

    ``main_token_ids[:, 0]`` is always committed.  The second main prediction is
    committed only when the draft equals the first prediction, matching the
    ``next_n=1`` reference algorithm.
    """
    main = main_token_ids.detach().cpu().to(torch.long)
    draft = draft_token_ids.detach().cpu().to(torch.long).reshape(-1)
    if main.ndim != 2 or main.shape[1] != 2:
        raise ValueError(f"main_token_ids must have shape [batch, 2], got {tuple(main.shape)}")
    if draft.numel() != main.shape[0]:
        raise ValueError(
            f"draft_token_ids must have {main.shape[0]} entries, got {draft.numel()}"
        )
    accepted: list[list[int]] = []
    for row in range(main.shape[0]):
        row_tokens = [int(main[row, 0].item())]
        if int(draft[row].item()) == row_tokens[0]:
            row_tokens.append(int(main[row, 1].item()))
        accepted.append(row_tokens)
    return accepted


class DeepSeekV4ModelRunner(ModelRunner):
    """Runner boundary for DeepSeekV4 W8A8 kernels and model-specific caches."""

    def __init__(self, *, compiled: DeepSeekV4CompiledKernels) -> None:
        super().__init__()
        self._compiled = compiled
        self.cache_manager = DeepSeekV4CacheManager(layout=compiled.layout)
        self.input_builder: DeepSeekV4InputBuilder | None = None
        self._l3_worker: Any | None = None
        self._l3_static_tensors: dict[
            tuple[int, tuple[int, ...], torch.dtype], StackedDeviceTensor
        ] = {}
        self._l3_cache_tensor_keys: set[tuple[int, tuple[int, ...], torch.dtype]] = set()
        self._decode_work_cache: DeepSeekV4LayerCache | None = None
        self._initialized_cache_requests: set[str] = set()
        self._global_weights: DeepSeekV4GlobalWeights | None = None
        self._static_final_norm_weight: torch.Tensor | None = None
        self._static_freqs_cos: torch.Tensor | None = None
        self._static_freqs_sin: torch.Tensor | None = None
        self._prefill_fwd_buffers: _DeepSeekV4PrefillFwdSharedBuffers | None = None
        self._decode_buffers: _DeepSeekV4DecodeSharedBuffers | None = None
        self._stacked_weight_buffers: dict[str, torch.Tensor] | None = None
        self._stacked_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._mtp_device_weights: dict[str, StackedDeviceTensor] | None = None
        self._hc_head_buffers: dict[str, torch.Tensor] | None = None
        self._decode_logits_buffer: torch.Tensor | None = None
        self._prefill_output_buffer: torch.Tensor | None = None
        self._prefill_pre_hc_output_buffer: torch.Tensor | None = None
        self._mtp_buffers: _DeepSeekV4MtpSharedBuffers | None = None
        self._mtp_prefill_context: DeepSeekV4PreparedPrefillInputs | None = None
        self._mtp_draft_token_ids: torch.Tensor | None = None
        self._mtp_tail_token_id: torch.Tensor | None = None
        self._mtp_tail_pre_hc_hidden: torch.Tensor | None = None
        self._mtp_tail_position: torch.Tensor | None = None
        self._mtp_initialized_slots: set[int] = set()
        self._mtp_proposed_tokens = 0
        self._mtp_accepted_tokens = 0

    def init_kv_cache(self, model_id: str, config: ModelConfig, runtime: RuntimeConfig) -> int:
        """Initialize runner state and return scheduler-only KV block capacity.

        DeepSeekV4 owns its NPU cache tensors and fixed slot mapping internally,
        so no generic KV tensors are allocated here. The scheduler still needs a
        positive block pool for host-side request budgeting and preemption.
        """
        self.input_builder = DeepSeekV4InputBuilder(
            layout=self._compiled.layout,
            hidden_size=config.hidden_size,
        )
        self._initialized_cache_requests.clear()
        if runtime.total_kv_pages is not None:
            return int(runtime.total_kv_pages)
        max_blocks_per_seq = math.ceil(runtime.max_seq_len / runtime.page_size)
        return int(runtime.max_batch_size * max_blocks_per_seq)

    def release_finished_requests(self, request_ids: Iterable[str]) -> None:
        """Release runner-owned cache slots for finished requests."""
        request_ids = tuple(request_ids)
        self.cache_manager.release(request_ids)
        if request_ids:
            if self._mtp_proposed_tokens:
                logger.info(
                    "DeepSeekV4 MTP acceptance: accepted=%d proposed=%d rate=%.2f%%",
                    self._mtp_accepted_tokens,
                    self._mtp_proposed_tokens,
                    100.0 * self._mtp_accepted_tokens / self._mtp_proposed_tokens,
                )
            self._initialized_cache_requests.difference_update(request_ids)
            # The host cache backing remains zeroed while kernels update the
            # resident copies. Drop those device copies when their only
            # supported serving request finishes, so a reused slot is uploaded
            # from clean backing storage instead of inheriting stale KV/state.
            if not self.cache_manager.active_slots:
                self._invalidate_resident_cache_tensors()
            self._mtp_prefill_context = None
            self._mtp_draft_token_ids = None
            self._mtp_tail_token_id = None
            self._mtp_tail_pre_hc_hidden = None
            self._mtp_tail_position = None
            self._mtp_initialized_slots.clear()
            self._mtp_proposed_tokens = 0
            self._mtp_accepted_tokens = 0

    def load_packed_global_weights(self) -> DeepSeekV4GlobalWeights:
        """Load global tensors and pack the LM head for host-side projection."""
        if self._global_weights is None:
            self._global_weights = self._compiled.weight_store.load_packed_global_weights(
                ranks=self._compiled.layout.ranks
            )
        return self._global_weights

    def load_stacked_layer_weights(self) -> DeepSeekV4StackedLayerWeights:
        """Load and stack all hidden-layer weights for the packed decode_fwd kernel."""
        compress_ratios = tuple(int(layer.compress_ratio) for layer in self._compiled.layer_plan)
        return self._compiled.weight_store.load_stacked_layer_weights(
            ranks=self._compiled.layout.ranks,
            n_routed_experts=self._compiled.n_routed_experts,
            compress_ratios=compress_ratios,
            num_hash_layers=self._compiled.num_hash_layers,
        )

    def load_mtp_weights(self) -> DeepSeekV4MtpWeights:
        """Load the single checkpoint MTP draft layer."""
        return self._compiled.weight_store.load_mtp_weights(
            ranks=self._compiled.layout.ranks,
            n_routed_experts=self._compiled.n_routed_experts,
        )

    def prepare_prefill_inputs(self, model: RuntimeModel, batch: PrefillBatch) -> DeepSeekV4PreparedPrefillInputs:
        """Build DeepSeekV4 prefill host inputs for the current scheduler chunk."""
        builder = self._require_input_builder()
        layout = self._compiled.layout
        if len(batch.request_ids) != layout.prefill_batch:
            raise ValueError(
                f"DeepSeekV4 prefill kernels support exactly {layout.prefill_batch} request per dispatch, "
                f"got {len(batch.request_ids)}"
            )
        request_id = batch.request_ids[0]
        slot = self.cache_manager.allocate(request_id)
        if slot is None:
            raise RuntimeError("DeepSeekV4 cache slots exhausted")

        actual_tokens = self._prefill_actual_tokens(batch)
        positions = self._prefill_positions(batch, actual_tokens)
        if positions[-1] >= model.runtime.max_seq_len:
            raise ValueError(
                f"prefill position {positions[-1]} exceeds max_seq_len={model.runtime.max_seq_len}"
            )
        if batch.input_embeddings is None:
            raise ValueError("DeepSeek V4 prefill requires host input embeddings")
        embeddings = batch.input_embeddings[0, :actual_tokens].to(torch.float32).cpu()
        token_ids = batch.token_ids[0, :actual_tokens].detach().cpu().to(torch.long)
        kernel_tokens = self._prefill_kernel_tokens(actual_tokens)
        kernel_positions = self._prefill_kernel_positions(
            positions,
            kernel_tokens=kernel_tokens,
            max_seq_len=model.runtime.max_seq_len,
        )
        kernel_slots = self._prefill_kernel_slots(
            slot,
            actual_tokens=actual_tokens,
            kernel_tokens=kernel_tokens,
        )
        kernel_embeddings = self._padded_rows(embeddings, kernel_tokens)
        kernel_token_ids = self._padded_vector(token_ids, kernel_tokens, dtype=torch.long)
        return DeepSeekV4PreparedPrefillInputs(
            request_id=request_id,
            slot=slot,
            actual_tokens=actual_tokens,
            x_hc=builder.prefill_x_hc(kernel_embeddings, actual_tokens=kernel_tokens),
            input_ids=self._rank_stack(self._padded_vector(kernel_token_ids, layout.prefill_seq, dtype=torch.long)),
            position_ids=self._rank_stack(self._prefill_position_ids(kernel_positions, layout.prefill_seq)),
            ori_block_table=self._rank_stack(
                self.cache_manager.block_table(
                    [slot],
                    max_blocks=layout.prefill_ori_max_blocks,
                    physical_blocks=layout.decode_ori_max_blocks,
                )[0]
            ),
            ori_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self._prefill_ori_slot_mapping(kernel_slots, kernel_positions),
                    layout.prefill_seq,
                )
            ),
            cmp_block_table=self._rank_stack(
                self.cache_manager.block_table([slot], max_blocks=layout.prefill_cmp_max_blocks)[0]
            ),
            idx_block_table=self._rank_stack(
                self.cache_manager.block_table([slot], max_blocks=layout.prefill_idx_max_blocks)[0]
            ),
            hca_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table(
                    [slot],
                    max_blocks=layout.prefill_hca_state_max_blocks,
                    physical_blocks=layout.hca_state_max_blocks,
                )[0]
            ),
            csa_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table(
                    [slot],
                    max_blocks=layout.prefill_csa_state_max_blocks,
                    physical_blocks=layout.csa_state_max_blocks,
                )[0]
            ),
            csa_inner_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table(
                    [slot],
                    max_blocks=layout.prefill_csa_inner_state_max_blocks,
                    physical_blocks=layout.csa_inner_state_max_blocks,
                )[0]
            ),
            hca_cmp_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self._prefill_compressed_slot_mapping(
                        kernel_slots,
                        kernel_positions,
                        max_blocks=layout.prefill_cmp_max_blocks,
                        compress_ratio=128,
                    ),
                    layout.prefill_seq,
                )
            ),
            hca_state_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self._prefill_state_slot_mapping(
                        kernel_slots,
                        kernel_positions,
                        max_blocks=layout.prefill_hca_state_max_blocks,
                        physical_blocks=layout.hca_state_max_blocks,
                        state_block_size=layout.c128_state_block_size,
                    ),
                    layout.prefill_seq,
                )
            ),
            csa_cmp_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self._prefill_compressed_slot_mapping(
                        kernel_slots,
                        kernel_positions,
                        max_blocks=layout.prefill_cmp_max_blocks,
                        compress_ratio=4,
                    ),
                    layout.prefill_seq,
                )
            ),
            csa_idx_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self._prefill_compressed_slot_mapping(
                        kernel_slots,
                        kernel_positions,
                        max_blocks=layout.prefill_idx_max_blocks,
                        compress_ratio=4,
                    ),
                    layout.prefill_seq,
                )
            ),
            csa_state_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self._prefill_state_slot_mapping(
                        kernel_slots,
                        kernel_positions,
                        max_blocks=layout.prefill_csa_state_max_blocks,
                        physical_blocks=layout.csa_state_max_blocks,
                        state_block_size=layout.c4_state_block_size,
                    ),
                    layout.prefill_seq,
                )
            ),
            csa_inner_state_slot_mapping=self._rank_stack(
                self._pad_prefill_mapping(
                    self._prefill_state_slot_mapping(
                        kernel_slots,
                        kernel_positions,
                        max_blocks=layout.prefill_csa_inner_state_max_blocks,
                        physical_blocks=layout.csa_inner_state_max_blocks,
                        state_block_size=layout.c4_state_block_size,
                    ),
                    layout.prefill_seq,
                )
            ),
        )

    def prepare_decode_inputs(self, model: RuntimeModel, batch: DecodeBatch) -> DeepSeekV4PreparedDecodeInputs:
        """Build DeepSeekV4 decode host inputs for the current scheduler batch."""
        builder = self._require_input_builder()
        layout = self._compiled.layout
        actual_batch = len(batch.request_ids)
        if actual_batch <= 0:
            raise ValueError("decode batch must contain at least one request")
        if actual_batch > layout.decode_batch:
            raise ValueError(f"decode batch {actual_batch} exceeds kernel batch {layout.decode_batch}")
        slots = self.cache_manager.slots_for_request_ids(batch.request_ids)
        positions = self._decode_positions(batch, actual_batch)
        max_position = max(max(row) for row in positions)
        if max_position >= model.runtime.max_seq_len:
            raise ValueError(f"decode position {max_position} exceeds max_seq_len={model.runtime.max_seq_len}")

        prev_token_ids = (
            batch.prev_token_ids.detach().cpu().to(torch.long)
            if batch.prev_token_ids is not None
            else None
        )
        token_ids = self._decode_token_rows(
            batch.token_ids.detach().cpu().to(torch.long),
            actual_batch,
            vocab_size=model.config.vocab_size,
            prev_token_ids=prev_token_ids,
        )
        decode_embeds = batch.hidden_states.to(torch.float32).cpu()
        prev_embeds = (
            batch.prev_hidden_states.to(torch.float32).cpu()
            if batch.prev_hidden_states is not None
            else None
        )
        if os.environ.get("PYPTO_DSV4_DIVERSE_DECODE_PAD") == "1" and actual_batch < layout.decode_batch:
            decode_embeds = self._diverse_decode_pad_embeddings(model, decode_embeds, actual_batch)
        x_hc = builder.decode_x_hc(decode_embeds, actual_batch=actual_batch, prev_embeddings=prev_embeds)
        decode_slots = self._decode_kernel_slots(slots)
        decode_positions = (*positions, *((positions[0],) * (layout.decode_batch - actual_batch)))
        ori_slot_mapping = self.cache_manager.sliding_window_slot_mapping(
            decode_slots,
            decode_positions,
            kernel_rows=layout.decode_batch,
        )
        # SWA layer: paged write position + full visible window (incl. current).
        swa_slot_mapping = self.cache_manager.paged_decode_slot_mapping(
            decode_slots,
            decode_positions,
            kernel_rows=layout.decode_batch,
        )
        swa_indices, swa_lens = self.cache_manager.swa_window_indices_and_lens(
            decode_slots,
            decode_positions,
            kernel_rows=layout.decode_batch,
        )
        # HCA/CSA use the same cache-first full window as pypto-lib: the current
        # decode-chunk KV rows are written to ori-KV before sparse attention.
        window_swa_indices, window_swa_lens = swa_indices, swa_lens
        hca_cmp_slot_mapping = self.cache_manager.compressed_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.cmp_max_blocks,
            compress_ratio=128,
            kernel_rows=layout.decode_batch,
        )
        hca_state_slot_mapping = self.cache_manager.state_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.prefill_hca_state_max_blocks,
            physical_blocks=layout.hca_state_max_blocks,
            state_block_size=layout.c128_state_block_size,
            kernel_rows=layout.decode_batch,
        )
        csa_cmp_slot_mapping = self.cache_manager.compressed_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.cmp_max_blocks,
            compress_ratio=4,
            kernel_rows=layout.decode_batch,
        )
        csa_idx_slot_mapping = self.cache_manager.compressed_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.idx_max_blocks,
            compress_ratio=4,
            kernel_rows=layout.decode_batch,
        )
        csa_state_slot_mapping = self.cache_manager.state_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.prefill_csa_state_max_blocks,
            physical_blocks=layout.csa_state_max_blocks,
            state_block_size=layout.c4_state_block_size,
            kernel_rows=layout.decode_batch,
        )
        csa_inner_state_slot_mapping = self.cache_manager.state_slot_mapping(
            decode_slots,
            decode_positions,
            max_blocks=layout.prefill_csa_inner_state_max_blocks,
            physical_blocks=layout.csa_inner_state_max_blocks,
            state_block_size=layout.c4_state_block_size,
            kernel_rows=layout.decode_batch,
        )

        return DeepSeekV4PreparedDecodeInputs(
            request_ids=tuple(batch.request_ids),
            slots=tuple(slots),
            kernel_slots=decode_slots,
            actual_batch=actual_batch,
            x_hc=x_hc,
            input_ids=self._rank_stack(token_ids),
            position_ids=self._rank_stack(torch.tensor(decode_positions, dtype=torch.int32).reshape(-1)),
            kv_seq_lens=self._rank_stack(self._decode_kv_seq_lens(batch.seq_lens, actual_batch)),
            block_table=self._rank_stack(
                self.cache_manager.paged_ori_block_table(
                    decode_slots,
                    kernel_rows=layout.decode_batch,
                )
            ),
            ori_slot_mapping=self._rank_stack(ori_slot_mapping.reshape(-1)),
            window_swa_indices=self._rank_stack(window_swa_indices),
            window_swa_lens=self._rank_stack(window_swa_lens),
            swa_slot_mapping=self._rank_stack(swa_slot_mapping.reshape(-1)),
            swa_indices=self._rank_stack(swa_indices),
            swa_lens=self._rank_stack(swa_lens),
            cmp_block_table=self._rank_stack(
                self.cache_manager.block_table_for_kernel_rows(
                    decode_slots,
                    max_blocks=layout.cmp_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
            ),
            idx_block_table=self._rank_stack(
                self.cache_manager.block_table_for_kernel_rows(
                    decode_slots,
                    max_blocks=layout.idx_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
            ),
            hca_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table_for_kernel_rows(
                    decode_slots,
                    max_blocks=layout.prefill_hca_state_max_blocks,
                    physical_blocks=layout.hca_state_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
            ),
            csa_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table_for_kernel_rows(
                    decode_slots,
                    max_blocks=layout.prefill_csa_state_max_blocks,
                    physical_blocks=layout.csa_state_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
            ),
            csa_inner_compress_state_block_table=self._rank_stack(
                self.cache_manager.block_table_for_kernel_rows(
                    decode_slots,
                    max_blocks=layout.prefill_csa_inner_state_max_blocks,
                    physical_blocks=layout.csa_inner_state_max_blocks,
                    kernel_rows=layout.decode_batch,
                )
            ),
            hca_cmp_slot_mapping=self._rank_stack(hca_cmp_slot_mapping.reshape(-1)),
            hca_state_slot_mapping=self._rank_stack(hca_state_slot_mapping.reshape(-1)),
            csa_cmp_slot_mapping=self._rank_stack(csa_cmp_slot_mapping.reshape(-1)),
            csa_idx_slot_mapping=self._rank_stack(csa_idx_slot_mapping.reshape(-1)),
            csa_state_slot_mapping=self._rank_stack(csa_state_slot_mapping.reshape(-1)),
            csa_inner_state_slot_mapping=self._rank_stack(csa_inner_state_slot_mapping.reshape(-1)),
        )

    def _diverse_decode_pad_embeddings(
        self, model, active_embeds: torch.Tensor, actual_batch: int
    ) -> torch.Tensor:
        """Diagnostic: build a full [decode_batch, hidden] embedding tensor whose
        padding rows carry distinct real token embeddings, so the decode MoE gate
        routes the 128 tokens across many experts instead of all padding rows
        mirroring slot 0. Active rows keep their real embeddings."""
        layout = self._compiled.layout
        embed = getattr(self, "_diverse_embed_cache", None)
        if embed is None:
            embed = self._compiled.weight_store.load_tensor("embed.weight").contiguous()
            self._diverse_embed_cache = embed
        vocab = int(model.config.vocab_size)
        hidden = int(embed.shape[1])
        full = torch.zeros((layout.decode_batch, hidden), dtype=active_embeds.dtype)
        full[:actual_batch].copy_(active_embeds[:actual_batch].to(full.dtype))
        # Distinct, spread-out, non-special token ids for the padding rows.
        pad_ids = [
            max(100, (1000 + row * 2659) % vocab)
            for row in range(actual_batch, layout.decode_batch)
        ]
        pad_embed = embed.index_select(0, torch.tensor(pad_ids, dtype=torch.long)).to(full.dtype)
        full[actual_batch:].copy_(pad_embed)
        return full

    def _alloc_kv_cache_tensor(self, shape: tuple[int, ...], dtype: torch.dtype) -> DeviceTensor:
        raise NotImplementedError("DeepSeekV4 uses model-specific cache pools, not generic KV tensors")

    def _free_kv_cache_tensor(self, tensor: DeviceTensor) -> None:
        return None

    def run_prefill(self, model, batch: PrefillBatch) -> PrefillResult:
        """Run all DeepSeekV4 hidden layers for one prefill chunk in a single packed call."""
        if self._compiled.prefill is None:
            raise RuntimeError("DeepSeekV4 kernels were not compiled for this runner")
        self._ensure_l3_shared_buffers(model)
        inputs = self.prepare_prefill_inputs(model, batch)
        if inputs.slot != 0:
            raise RuntimeError(
                "DeepSeekV4 prefill currently supports the first active serving slot only. "
                "Run with one concurrent request until pypto-lib exposes a 64-slot prefill kernel."
            )
        self._stage_prefill_fwd_inputs(inputs)
        hidden_buffer = self._require_prefill_output_buffer(model.config.hidden_size)
        pre_hc_hidden_buffer = self._require_prefill_pre_hc_output_buffer(model.config.hidden_size)
        hidden_buffer.zero_()
        pre_hc_hidden_buffer.zero_()
        args = self._prefill_fwd_args(pre_hc_hidden_buffer, hidden_buffer)
        self._debug_prefill_dispatch(inputs, args)
        if os.environ.get("PYPTO_DSV4_SKIP_PREFILL_KERNEL") == "1":
            # Diagnostic: skip the prefill kernel dispatch to isolate the decode
            # deadlock from prefill device/ring state. All host-side prep above
            # ran; we only skip the device kernel. Snapshot the (un-run) packed
            # unified caches so decode can proceed, and force the first token to " a"
            # (id 260) so decode runs on a realistic input.
            forced = torch.zeros((1, int(model.config.vocab_size)), dtype=torch.float32)
            forced[0, 260] = 1.0e4
            return PrefillResult(last_hidden=None, logits=forced)
        try:
            self._run_l3(
                self._require_prefill_callable(),
                *args,
                self._int32_scalar(self._prefill_kernel_tokens(inputs.actual_tokens)),
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "DeepSeekV4 packed prefill dispatch failed "
                f"(tokens={inputs.actual_tokens}, slot={inputs.slot})"
            ) from exc
        if self._mtp_buffers is not None:
            # MTP prefill is shifted by one token and therefore needs the first
            # sampled token.  Keep the immutable host metadata here; the first
            # decode call supplies that token and completes draft initialization.
            self._mtp_prefill_context = inputs

        active_hidden = hidden_buffer[:, : inputs.actual_tokens, :]
        self._debug_tensor_stats("prefill.output.hidden.active", active_hidden, per_rank=True)
        if self._debug_tensor_stats_enabled() and not self._tensor_is_finite(active_hidden):
            raise RuntimeError("DeepSeekV4 packed prefill produced non-finite active hidden rows")

        # Sample the last real prompt row (``actual_tokens - 1``) from host-side
        # LM-head logits, mirroring the decode path.
        last_row = inputs.actual_tokens - 1
        logits = self._logits_for_hidden(hidden_buffer, active_rows=(last_row,), label="prefill").float()
        return PrefillResult(last_hidden=None, logits=logits)

    def run_decode(self, model, batch: DecodeBatch) -> DecodeResult:
        """Run all DeepSeekV4 hidden layers for one decode batch in a single packed call."""
        if self._compiled.decode is None:
            raise RuntimeError("DeepSeekV4 kernels were not compiled for this runner")
        self._ensure_l3_shared_buffers(model)
        mtp_active = self._mtp_buffers is not None and batch.allow_device_greedy_sampling
        if mtp_active and self._mtp_draft_token_ids is None and self._mtp_prefill_context is not None:
            self._initialize_mtp_draft(model, batch)
        speculative_step = mtp_active and self._mtp_draft_token_ids is not None
        if speculative_step:
            batch = self._main_speculative_batch(model, batch, self._mtp_draft_token_ids)
        inputs = self._stage_decode_inputs(self.prepare_decode_inputs(model, batch))
        if inputs.actual_batch != 1 or inputs.slots != (0,):
            raise RuntimeError(
                "DeepSeekV4 decode currently supports the first active serving slot only. "
                "Run with one concurrent request until the compact cache handoff supports multiple slots."
            )
        decode_buffers = self._require_decode_buffers()
        x_hc = decode_buffers.x_hc_a
        active_decode_tokens = inputs.actual_batch * self._compiled.layout.decode_seq
        self._debug_tensor_stats("decode.input.initial.active", x_hc[:, :active_decode_tokens, :, :])

        hidden_buffer = self._require_decode_output_buffer(model.config.hidden_size)
        pre_hc_hidden_buffer = decode_buffers.pre_hc_hidden_out
        hidden_buffer.zero_()
        pre_hc_hidden_buffer.zero_()
        # ``num_tokens`` is the real active token count. PR 677 restores the
        # gate norm/quant scopes for this num_tokens-aware path; the fixed padding
        # rows remain valid metadata for attention but must not be routed by MoE.
        num_tokens = active_decode_tokens
        args = self._decode_fwd_args(inputs, x_hc, pre_hc_hidden_buffer, hidden_buffer)
        self._debug_decode_dispatch(inputs, args)
        try:
            self._run_l3(
                self._require_decode_callable(),
                *args,
                self._int32_scalar(num_tokens),
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "DeepSeekV4 packed decode dispatch failed "
                f"(actual_batch={inputs.actual_batch}, slots={inputs.slots})"
            ) from exc
        active_hidden = hidden_buffer[:, :active_decode_tokens, :]
        self._debug_tensor_stats("decode.output.hidden.active", active_hidden, per_rank=True)
        if self._debug_tensor_stats_enabled() and not self._tensor_is_finite(active_hidden):
            raise RuntimeError("DeepSeekV4 packed decode produced non-finite active hidden rows")

        decode_seq = self._compiled.layout.decode_seq
        if speculative_step:
            rows = tuple(range(inputs.actual_batch * decode_seq))
            pair_logits = self._logits_for_hidden(hidden_buffer, active_rows=rows, label="decode.mtp_main").float()
            main_ids = pair_logits.argmax(dim=-1).reshape(inputs.actual_batch, decode_seq)
            accepted = accept_mtp_tokens(main_ids, self._mtp_draft_token_ids[: inputs.actual_batch])
            self._mtp_proposed_tokens += inputs.actual_batch
            self._mtp_accepted_tokens += sum(len(tokens) == decode_seq for tokens in accepted)
            logger.info(
                "DeepSeekV4 MTP acceptance progress: accepted=%d proposed=%d rate=%.2f%%",
                self._mtp_accepted_tokens,
                self._mtp_proposed_tokens,
                100.0 * self._mtp_accepted_tokens / self._mtp_proposed_tokens,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "DeepSeekV4 MTP step: draft=%s main=%s",
                    self._mtp_draft_token_ids[: inputs.actual_batch].detach().cpu().tolist(),
                    main_ids.detach().cpu().tolist(),
                )
            # Match the reference accepted_num flow: update the MTP window from
            # committed main-model outputs immediately, even after rejection.
            # On rejection only row 0 is accepted; the helper prepends the last
            # committed MTP row instead of consuming row 1, whose hidden state
            # depends on the rejected draft. The next cache-first main decode
            # overwrites the rejected position with the committed token.
            self._advance_mtp_draft(
                model,
                inputs,
                main_ids,
                pre_hc_hidden_buffer,
                accepted_count=len(accepted[0]),
            )
            return DecodeResult(
                hidden_states=None,
                logits=pair_logits[::decode_seq],
                accepted_token_ids=accepted,
            )

        # Standard B4S2 decode samples only the final real slot.
        active_rows = tuple(row * decode_seq + (decode_seq - 1) for row in range(inputs.actual_batch))
        logits = self._logits_for_hidden(hidden_buffer, active_rows=active_rows, label="decode").float()
        return DecodeResult(hidden_states=None, logits=logits)

    def _require_prefill_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.prefill is None:
            raise RuntimeError("DeepSeekV4 prefill kernel is not compiled")
        return self._compiled.prefill

    def _require_decode_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.decode is None:
            raise RuntimeError("DeepSeekV4 decode kernel is not compiled")
        return self._compiled.decode

    def _ensure_l3_shared_buffers(self, model: RuntimeModel) -> None:
        """Allocate every CPU tensor visible to the L3 worker before it forks.

        ``DistributedWorker`` creates per-chip children on first use. Any CPU
        tensor argument those children access must already live in shared memory
        at that point, so this method stages all packed prefill/decode input and
        output buffers before the first ``_run_l3`` call.
        """
        self.load_packed_global_weights()
        self._static_freqs_cos_tensor()
        self._static_freqs_sin_tensor()
        self._ensure_decode_buffers(model.config.hidden_size)
        self._ensure_mtp_buffers(model.config.hidden_size)
        self._ensure_decode_work_cache()
        self._require_prefill_output_buffer(model.config.hidden_size)
        self._require_prefill_pre_hc_output_buffer(model.config.hidden_size)
        self._static_final_norm_weight_tensor()
        if self._stacked_weight_buffers is None:
            if self._stacked_device_weights is None:
                self._stage_stacked_weights(self.load_stacked_layer_weights())
        self._hc_head_tensors()
        self._ensure_prefill_fwd_buffers(model.config.hidden_size)
        self._assert_l3_shared_buffers_preallocated()
        self._materialize_resident_weights()

    def _assert_l3_shared_buffers_preallocated(self) -> None:
        missing = self._missing_l3_shared_buffers()
        if missing:
            raise RuntimeError(
                "DeepSeekV4 L3 worker cannot start before all shared host buffers are preallocated; "
                "missing: " + ", ".join(missing)
            )

    def _missing_l3_shared_buffers(self) -> list[str]:
        missing: list[str] = []
        expected = {
            "final_norm_w": self._static_final_norm_weight,
            "freqs_cos": self._static_freqs_cos,
            "freqs_sin": self._static_freqs_sin,
            "prefill_fwd_buffers": self._prefill_fwd_buffers,
            "decode_buffers": self._decode_buffers,
            "decode_work_cache": self._decode_work_cache,
            "stacked_weights": self._stacked_weight_buffers or self._stacked_device_weights,
            "hc_head_buffers": self._hc_head_buffers,
            "prefill_output": self._prefill_output_buffer,
            "prefill_pre_hc_output": self._prefill_pre_hc_output_buffer,
        }
        if self._compiled.mtp_prefill is not None or self._compiled.mtp_decode is not None:
            expected["mtp_buffers"] = self._mtp_buffers
        for name, value in expected.items():
            if value is None:
                missing.append(name)
        if self._stacked_weight_buffers is not None and not self._stacked_weight_buffers:
            missing.append("stacked_weights")
        if self._hc_head_buffers is not None and not self._hc_head_buffers:
            missing.append("hc_head_buffers")
        return missing

    def _prefill_fwd_args(
        self,
        pre_hc_hidden_out: torch.Tensor,
        x_out: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Build the single packed ``l3_prefill_fwd`` argument tuple.

        The kernel runs final RMSNorm and emits normalized hidden rows. LM-head is
        computed on the host from the selected rows.
        """
        buffers = self._require_prefill_fwd_buffers()
        stacked = self._require_stacked_weights()
        hc_head = self._hc_head_tensors()
        values = dict(stacked.tensors)
        values.update(
            {
                "x_hc": buffers.x_hc,
                "freqs_cos": buffers.freqs_cos,
                "freqs_sin": buffers.freqs_sin,
                "hc_head_fn": hc_head["hc_head_fn"],
                "hc_head_scale": hc_head["hc_head_scale"],
                "hc_head_base": hc_head["hc_head_base"],
                "final_norm_w": self._static_final_norm_weight_tensor(),
                "pre_hc_hidden_out": pre_hc_hidden_out,
                "x_out": x_out,
            }
        )
        values.update(buffers.tensors)
        values = self._mark_resident_args(values, _PREFILL_RESIDENT_POLICY)
        return self._ordered_layer_args(values, _PREFILL_FWD_TENSOR_ORDER)

    def _decode_fwd_args(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        x_hc: torch.Tensor,
        pre_hc_hidden_out: torch.Tensor,
        x_out: torch.Tensor,
    ) -> tuple[Any, ...]:
        """Build the single packed ``l3_decode_fwd`` argument tuple."""
        cache = self._require_decode_work_cache()
        stacked = self._require_stacked_weights()
        hc_head = self._hc_head_tensors()
        values = dict(stacked.tensors)
        values.update(
            {
                "x_hc": x_hc,
                "freqs_cos": self._static_freqs_cos_tensor(),
                "freqs_sin": self._static_freqs_sin_tensor(),
                "kv_cache": cache.kv_cache,
                "block_table": inputs.block_table,
                "ori_slot_mapping": inputs.ori_slot_mapping,
                "window_swa_indices": inputs.window_swa_indices,
                "window_swa_lens": inputs.window_swa_lens,
                "swa_slot_mapping": inputs.swa_slot_mapping,
                "swa_indices": inputs.swa_indices,
                "swa_lens": inputs.swa_lens,
                "hca_cmp_slot_mapping": inputs.hca_cmp_slot_mapping,
                "hca_state_slot_mapping": inputs.hca_state_slot_mapping,
                "csa_cmp_slot_mapping": inputs.csa_cmp_slot_mapping,
                "csa_idx_slot_mapping": inputs.csa_idx_slot_mapping,
                "csa_state_slot_mapping": inputs.csa_state_slot_mapping,
                "csa_inner_state_slot_mapping": inputs.csa_inner_state_slot_mapping,
                "position_ids": inputs.position_ids,
                "kv_seq_lens": inputs.kv_seq_lens,
                "hca_compress_state": cache.hca_compress_state,
                "hca_compress_state_block_table": inputs.hca_compress_state_block_table,
                "csa_compress_state": cache.csa_compress_state,
                "csa_compress_state_block_table": inputs.csa_compress_state_block_table,
                "csa_inner_compress_state": cache.csa_inner_compress_state,
                "csa_inner_compress_state_block_table": inputs.csa_inner_compress_state_block_table,
                "cmp_kv": cache.cmp_kv,
                "cmp_block_table": inputs.cmp_block_table,
                "idx_kv_cache": cache.idx_kv_cache,
                "idx_kv_scale": cache.idx_kv_scale,
                "idx_block_table": inputs.idx_block_table,
                "input_ids": inputs.input_ids,
                "hc_head_fn": hc_head["hc_head_fn"],
                "hc_head_scale": hc_head["hc_head_scale"],
                "hc_head_base": hc_head["hc_head_base"],
                "final_norm_w": self._static_final_norm_weight_tensor(),
                "pre_hc_hidden_out": pre_hc_hidden_out,
                "x_out": x_out,
            }
        )
        values = self._mark_resident_args(values, _DECODE_RESIDENT_POLICY)
        return self._ordered_layer_args(values, _DECODE_FWD_TENSOR_ORDER)

    def _mtp_prefill_args(self) -> tuple[Any, ...]:
        buffers = self._require_mtp_buffers()
        values = dict(self._mtp_device_weights or buffers.weights)
        values.update(
            {
                "hidden_states": buffers.prefill_hidden_in,
                "prev_hidden_states": buffers.prefill_prev_hidden_in,
                "freqs_cos": self._static_freqs_cos_tensor(),
                "freqs_sin": self._static_freqs_sin_tensor(),
                "kv_cache": buffers.prefill_kv_cache,
                "ori_block_table": buffers.prefill_block_table,
                "ori_slot_mapping": buffers.prefill_slot_mapping,
                "position_ids": buffers.prefill_position_ids,
                "input_ids": buffers.prefill_input_ids,
                "hidden_out": buffers.prefill_hidden_out,
                "pre_hc_hidden_out": buffers.prefill_pre_hc_out,
            }
        )
        values = self._mark_resident_args(values, _MTP_RESIDENT_POLICY)
        return self._ordered_layer_args(values, _MTP_PREFILL_TENSOR_ORDER)

    def _mtp_decode_args(self) -> tuple[Any, ...]:
        buffers = self._require_mtp_buffers()
        values = dict(self._mtp_device_weights or buffers.weights)
        values.update(
            {
                "hidden_states": buffers.decode_hidden_in,
                "prev_pre_hc_hidden": buffers.decode_prev_hidden_in,
                "position_ids": buffers.decode_position_ids,
                "freqs_cos": self._static_freqs_cos_tensor(),
                "freqs_sin": self._static_freqs_sin_tensor(),
                "kv_cache": buffers.decode_kv_cache,
                "swa_slot_mapping": buffers.decode_slot_mapping,
                "swa_indices": buffers.decode_swa_indices,
                "swa_lens": buffers.decode_swa_lens,
                "input_ids": buffers.decode_input_ids,
                "hidden_out": buffers.decode_hidden_out,
                "next_pre_hc_hidden": buffers.decode_pre_hc_out,
            }
        )
        values = self._mark_resident_args(values, _MTP_RESIDENT_POLICY)
        return self._ordered_layer_args(values, _MTP_DECODE_TENSOR_ORDER)

    def _require_mtp_buffers(self) -> _DeepSeekV4MtpSharedBuffers:
        if self._mtp_buffers is None:
            raise RuntimeError("DeepSeekV4 MTP shared buffers are not staged")
        return self._mtp_buffers

    def _require_mtp_prefill_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.mtp_prefill is None:
            raise RuntimeError("DeepSeekV4 MTP prefill kernel is not compiled")
        return self._compiled.mtp_prefill

    def _require_mtp_decode_callable(self) -> DeepSeekV4L3Callable:
        if self._compiled.mtp_decode is None:
            raise RuntimeError("DeepSeekV4 MTP decode kernel is not compiled")
        return self._compiled.mtp_decode

    def _embedding_rows(self, token_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        embed = self._compiled.embedding_weight
        if embed is None:
            embed = self._compiled.weight_store.load_tensor("embed.weight").contiguous().cpu()
            self._compiled.embedding_weight = embed
        return embed.index_select(0, token_ids.detach().cpu().to(torch.long).reshape(-1)).to(dtype)

    def _main_speculative_batch(
        self,
        model: RuntimeModel,
        batch: DecodeBatch,
        draft_token_ids: torch.Tensor,
    ) -> DecodeBatch:
        actual_batch = len(batch.request_ids)
        draft = draft_token_ids[:actual_batch].detach().cpu().to(torch.long)
        current = batch.token_ids[:actual_batch].detach().cpu().to(torch.long).reshape(-1)
        current_hidden = batch.hidden_states[:actual_batch].detach().cpu()
        draft_hidden = self._embedding_rows(draft, current_hidden.dtype)
        return replace(
            batch,
            token_ids=draft.reshape(actual_batch, 1),
            hidden_states=draft_hidden,
            prev_token_ids=current,
            prev_hidden_states=current_hidden,
            seq_lens=batch.seq_lens.detach().cpu().to(torch.int32) + 1,
        )

    def _initialize_mtp_draft(self, model: RuntimeModel, batch: DecodeBatch) -> None:
        context = self._mtp_prefill_context
        if context is None:
            return
        if len(batch.request_ids) != 1 or context.request_id != batch.request_ids[0]:
            raise RuntimeError("DeepSeekV4 MTP prefill context does not match the decode request")
        buffers = self._require_mtp_buffers()
        n = context.actual_tokens
        first_token = batch.token_ids[0].detach().cpu().to(torch.long).reshape(1)
        first_hidden = batch.hidden_states[0].detach().cpu().to(torch.bfloat16)
        buffers.prefill_hidden_in.zero_()
        buffers.prefill_hidden_in[:, : n - 1].copy_(context.x_hc[:, 1:n, 0].to(torch.bfloat16))
        buffers.prefill_hidden_in[:, n - 1].copy_(first_hidden)
        buffers.prefill_prev_hidden_in.zero_()
        buffers.prefill_prev_hidden_in[:, :n].copy_(
            self._require_prefill_pre_hc_output_buffer(model.config.hidden_size)[:, :n]
        )
        buffers.prefill_input_ids.copy_(context.input_ids)
        buffers.prefill_input_ids[:, : n - 1].copy_(context.input_ids[:, 1:n])
        buffers.prefill_input_ids[:, n - 1].copy_(first_token)
        buffers.prefill_position_ids.copy_(context.position_ids)
        buffers.prefill_block_table.copy_(context.ori_block_table)
        buffers.prefill_slot_mapping.copy_(context.ori_slot_mapping)
        buffers.prefill_hidden_out.zero_()
        buffers.prefill_pre_hc_out.zero_()
        self._run_l3(
            self._require_mtp_prefill_callable(),
            *self._mtp_prefill_args(),
            self._int32_scalar(n),
        )
        draft_logits = self._logits_for_hidden(
            buffers.prefill_hidden_out,
            active_rows=(n - 1,),
            label="mtp.prefill",
        )
        self._mtp_draft_token_ids = draft_logits.argmax(dim=-1).to(torch.long)
        # The shifted MTP row for ``first_token`` is paired with the main-model
        # hidden state and position of the prompt's final token. Keep that tail
        # so a later rejection can rebuild the fixed two-row MTP window from
        # [last committed token, newly accepted token], as the reference runner
        # does when slicing prev_hidden_states to ``spec_len``.
        self._mtp_tail_token_id = first_token.clone()
        self._mtp_tail_pre_hc_hidden = buffers.prefill_prev_hidden_in[:, n - 1].clone()
        self._mtp_tail_position = context.position_ids[0, n - 1].reshape(1).clone()
        self._mtp_initialized_slots.add(context.slot)
        self._mtp_prefill_context = None

    def _mtp_committed_window(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        main_ids: torch.Tensor,
        main_pre_hc: torch.Tensor,
        *,
        accepted_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the fixed MTP window from accepted main-model outputs only."""
        layout = self._compiled.layout
        if inputs.actual_batch != 1:
            raise RuntimeError("DeepSeekV4 MTP accepted-window update requires one active request")
        if accepted_count == layout.decode_seq:
            return (
                main_ids[0, :layout.decode_seq].detach().cpu().to(torch.long),
                main_pre_hc[:, :layout.decode_seq].detach().cpu(),
                inputs.position_ids[0, :layout.decode_seq].detach().cpu().to(torch.int32),
            )
        if accepted_count != 1 or layout.decode_seq != 2:
            raise ValueError(
                "DeepSeekV4 MTP currently supports next_n=1 acceptance only; "
                f"got accepted_count={accepted_count}, decode_seq={layout.decode_seq}"
            )
        if (
            self._mtp_tail_token_id is None
            or self._mtp_tail_pre_hc_hidden is None
            or self._mtp_tail_position is None
        ):
            raise RuntimeError("DeepSeekV4 MTP committed tail is not initialized")
        committed_ids = torch.cat(
            (
                self._mtp_tail_token_id.reshape(1),
                main_ids[0, :1].detach().cpu().to(torch.long),
            )
        )
        committed_pre_hc = torch.cat(
            (
                self._mtp_tail_pre_hc_hidden.unsqueeze(1),
                main_pre_hc[:, :1].detach().cpu(),
            ),
            dim=1,
        )
        committed_positions = torch.cat(
            (
                self._mtp_tail_position.reshape(1),
                inputs.position_ids[0, :1].detach().cpu().to(torch.int32),
            )
        )
        return committed_ids, committed_pre_hc, committed_positions

    def _advance_mtp_draft(
        self,
        model: RuntimeModel,
        inputs: DeepSeekV4PreparedDecodeInputs,
        main_ids: torch.Tensor,
        main_pre_hc: torch.Tensor,
        *,
        accepted_count: int,
    ) -> None:
        """Advance the MTP draft window from tokens committed by the main model."""
        del model
        buffers = self._require_mtp_buffers()
        layout = self._compiled.layout
        active_tokens = inputs.actual_batch * layout.decode_seq
        committed_pair, committed_pre_hc, mtp_positions = self._mtp_committed_window(
            inputs,
            main_ids,
            main_pre_hc,
            accepted_count=accepted_count,
        )

        kernel_ids = committed_pair.repeat(layout.decode_batch)
        embeddings = self._embedding_rows(kernel_ids, torch.bfloat16)
        buffers.decode_hidden_in.zero_()
        buffers.decode_hidden_in.copy_(embeddings.unsqueeze(0).expand(layout.ranks, -1, -1))
        buffers.decode_prev_hidden_in.zero_()
        buffers.decode_prev_hidden_in[:, :active_tokens].copy_(committed_pre_hc[:, :active_tokens])
        if active_tokens < layout.decode_tokens:
            buffers.decode_prev_hidden_in[:, active_tokens:].copy_(
                committed_pre_hc[:, :1].expand(-1, layout.decode_tokens - active_tokens, -1, -1)
            )
        buffers.decode_input_ids.copy_(self._rank_stack(kernel_ids))

        decode_positions = tuple(
            tuple(int(position) for position in mtp_positions.tolist())
            for _ in range(layout.decode_batch)
        )
        buffers.decode_position_ids.copy_(
            self._rank_stack(torch.tensor(decode_positions, dtype=torch.int32).reshape(-1))
        )
        slot_mapping = self.cache_manager.paged_decode_slot_mapping(
            inputs.kernel_slots,
            decode_positions,
            kernel_rows=layout.decode_batch,
        )
        swa_indices, swa_lens = self.cache_manager.swa_window_indices_and_lens(
            inputs.kernel_slots,
            decode_positions,
            kernel_rows=layout.decode_batch,
        )
        buffers.decode_slot_mapping.copy_(self._rank_stack(slot_mapping.reshape(-1)))
        buffers.decode_swa_indices.copy_(self._rank_stack(swa_indices))
        buffers.decode_swa_lens.copy_(self._rank_stack(swa_lens))
        buffers.decode_hidden_out.zero_()
        buffers.decode_pre_hc_out.zero_()
        self._run_l3(
            self._require_mtp_decode_callable(),
            *self._mtp_decode_args(),
            self._int32_scalar(active_tokens),
        )
        draft_rows = tuple(
            row * layout.decode_seq + layout.decode_seq - 1 for row in range(inputs.actual_batch)
        )
        draft_logits = self._logits_for_hidden(
            buffers.decode_hidden_out,
            active_rows=draft_rows,
            label="mtp.decode",
        )
        self._mtp_draft_token_ids = draft_logits.argmax(dim=-1).to(torch.long)
        self._mtp_tail_token_id = committed_pair[-1:].clone()
        self._mtp_tail_pre_hc_hidden = committed_pre_hc[:, -1].clone()
        self._mtp_tail_position = mtp_positions[-1:].clone()

    def _require_stacked_weights(self) -> DeepSeekV4StackedLayerWeights:
        tensors = self._stacked_device_weights or self._stacked_weight_buffers
        if tensors is None:
            raise RuntimeError("DeepSeekV4 stacked decode weights were not staged")
        return DeepSeekV4StackedLayerWeights(tensors=tensors)

    def _ordered_layer_args(self, values: dict[str, Any], names: Sequence[str]) -> tuple[Any, ...]:
        missing = [name for name in names if name not in values]
        if missing:
            raise KeyError(f"DeepSeekV4 layer dispatch is missing tensors: {', '.join(missing)}")
        return tuple(values[name] for name in names)

    def _debug_prefill_dispatch(
        self,
        inputs: DeepSeekV4PreparedPrefillInputs,
        args: Sequence[Any],
    ) -> None:
        if os.getenv("PYPTO_DSV4_DEBUG") != "1":
            return
        named_args = dict(zip(_PREFILL_FWD_TENSOR_ORDER, args, strict=True))
        interesting = (
            "x_hc",
            "kv_cache",
            "cmp_kv",
            "idx_kv_cache",
            "ori_block_table",
            "cmp_block_table",
            "idx_block_table",
            "input_ids",
            "x_out",
        )
        tensor_names = [
            name
            for name, tensor in named_args.items()
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
        ]
        non_shared = [name for name in tensor_names if not named_args[name].is_shared()]
        parts = []
        for name in interesting:
            tensor = named_args[name]
            if isinstance(tensor, torch.Tensor):
                parts.append(f"{name}={tuple(tensor.shape)}/{tensor.dtype}/shared={tensor.is_shared()}")
            elif isinstance(tensor, DeviceTensor):
                parts.append(f"{name}=DeviceTensor")
            else:
                parts.append(f"{name}={type(tensor).__name__}")
        print(
            "DeepSeekV4 packed prefill dispatch "
            f"tokens={inputs.actual_tokens} slot={inputs.slot} "
            f"worker_started={self._l3_worker is not None} "
            f"cpu_tensor_args={len(tensor_names)} non_shared={non_shared} "
            + " ".join(parts),
            flush=True,
        )
        if os.getenv("PYPTO_DSV4_DEBUG_ARGS") == "1":
            for name in _PREFILL_FWD_TENSOR_ORDER:
                tensor = named_args[name]
                if isinstance(tensor, torch.Tensor):
                    print(
                        "DeepSeekV4 prefill arg "
                        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                        f"device={tensor.device} shared={tensor.is_shared()}",
                        flush=True,
                    )

    def _debug_decode_dispatch(
        self,
        inputs: DeepSeekV4PreparedDecodeInputs,
        args: Sequence[Any],
    ) -> None:
        if os.getenv("PYPTO_DSV4_DEBUG") != "1":
            return
        named_args = dict(zip(_DECODE_FWD_TENSOR_ORDER, args, strict=True))
        interesting = (
            "x_hc",
            "kv_cache",
            "block_table",
            "ori_slot_mapping",
            "cmp_kv",
            "cmp_block_table",
            "idx_kv_cache",
            "idx_block_table",
            "hca_compress_state",
            "hca_state_slot_mapping",
            "csa_compress_state",
            "csa_state_slot_mapping",
            "csa_inner_compress_state",
            "csa_inner_state_slot_mapping",
            "position_ids",
            "kv_seq_lens",
            "input_ids",
            "x_out",
        )
        tensor_names = [
            name
            for name, tensor in named_args.items()
            if isinstance(tensor, torch.Tensor) and tensor.device.type == "cpu"
        ]
        non_shared = [name for name in tensor_names if not named_args[name].is_shared()]
        parts = []
        for name in interesting:
            tensor = named_args[name]
            if isinstance(tensor, torch.Tensor):
                parts.append(f"{name}={tuple(tensor.shape)}/{tensor.dtype}/shared={tensor.is_shared()}")
            elif isinstance(tensor, DeviceTensor):
                parts.append(f"{name}=DeviceTensor")
            else:
                parts.append(f"{name}={type(tensor).__name__}")
        print(
            "DeepSeekV4 packed decode dispatch "
            f"actual_batch={inputs.actual_batch} active_tokens={inputs.actual_batch * self._compiled.layout.decode_seq} "
            f"slots={inputs.slots} "
            f"worker_started={self._l3_worker is not None} "
            f"cpu_tensor_args={len(tensor_names)} non_shared={non_shared} "
            + " ".join(parts),
            flush=True,
        )
        if os.getenv("PYPTO_DSV4_DEBUG_ARGS") == "1":
            for name in _DECODE_FWD_TENSOR_ORDER:
                tensor = named_args[name]
                if isinstance(tensor, torch.Tensor):
                    print(
                        "DeepSeekV4 decode arg "
                        f"{name}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                        f"device={tensor.device} shared={tensor.is_shared()}",
                        flush=True,
                    )
                    self._debug_tensor_stats(f"dispatch.fwd.{name}", tensor)

    @staticmethod
    def _is_layer_weight_name(name: str) -> bool:
        runtime_names = {
            "x_hc",
            "freqs_cos",
            "freqs_sin",
            "hca_compress_state_block_table",
            "csa_compress_state_block_table",
            "csa_inner_compress_state_block_table",
            "kv_cache",
            "ori_block_table",
            "block_table",
            "ori_slot_mapping",
            "cmp_kv",
            "cmp_block_table",
            "idx_kv_cache",
            "idx_kv_scale",
            "idx_block_table",
            "position_ids",
            "window_swa_indices",
            "window_swa_lens",
            "swa_slot_mapping",
            "swa_indices",
            "swa_lens",
            "hca_cmp_slot_mapping",
            "hca_state_slot_mapping",
            "csa_cmp_slot_mapping",
            "csa_idx_slot_mapping",
            "csa_state_slot_mapping",
            "csa_inner_state_slot_mapping",
            "hca_compress_state",
            "csa_compress_state",
            "csa_inner_compress_state",
            "kv_seq_lens",
            "input_ids",
            "x_next",
        }
        return name not in runtime_names

    def _ensure_decode_buffers(self, hidden_size: int) -> _DeepSeekV4DecodeSharedBuffers:
        buffers = self._decode_buffers
        if buffers is None:
            self._ensure_shared_host_allocation_before_worker("decode inputs")
            layout = self._compiled.layout
            ranks = layout.ranks
            batch = layout.decode_batch
            tokens = layout.decode_tokens
            buffers = _DeepSeekV4DecodeSharedBuffers(
                x_hc_a=self._shared_empty(
                    (ranks, tokens, layout.hc_mult, int(hidden_size)),
                    torch.float32,
                    name="decode_x_hc",
                ),
                x_hc_b=self._shared_empty(
                    (ranks, tokens, layout.hc_mult, int(hidden_size)),
                    torch.float32,
                    name="decode_x_hc_next",
                ),
                pre_hc_hidden_out=self._shared_empty(
                    (ranks, tokens, layout.hc_mult, int(hidden_size)),
                    torch.float32,
                    name="decode_pre_hc_hidden_out",
                ),
                x_out=self._shared_empty(
                    (ranks, tokens, int(hidden_size)),
                    torch.bfloat16,
                    name="decode_x_out",
                ),
                tensors={
                    "input_ids": self._shared_empty((ranks, tokens), torch.long, name="decode_input_ids"),
                    "position_ids": self._shared_empty((ranks, tokens), torch.int32, name="decode_position_ids"),
                    "kv_seq_lens": self._shared_empty((ranks, batch), torch.int32, name="decode_kv_seq_lens"),
                    "block_table": self._shared_empty(
                        (ranks, batch, layout.ori_table_max_blocks),
                        torch.int32,
                        name="decode_block_table",
                    ),
                    "ori_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_ori_slot_mapping",
                    ),
                    "window_swa_indices": self._shared_empty(
                        (ranks, tokens, layout.sliding_window),
                        torch.int32,
                        name="decode_window_swa_indices",
                    ),
                    "window_swa_lens": self._shared_empty(
                        (ranks, tokens),
                        torch.int32,
                        name="decode_window_swa_lens",
                    ),
                    "swa_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_swa_slot_mapping",
                    ),
                    "swa_indices": self._shared_empty(
                        (ranks, tokens, layout.sliding_window),
                        torch.int32,
                        name="decode_swa_indices",
                    ),
                    "swa_lens": self._shared_empty(
                        (ranks, tokens),
                        torch.int32,
                        name="decode_swa_lens",
                    ),
                    "cmp_block_table": self._shared_empty(
                        (ranks, batch, layout.cmp_max_blocks),
                        torch.int32,
                        name="decode_cmp_block_table",
                    ),
                    "idx_block_table": self._shared_empty(
                        (ranks, batch, layout.idx_max_blocks),
                        torch.int32,
                        name="decode_idx_block_table",
                    ),
                    "hca_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.prefill_hca_state_max_blocks),
                        torch.int32,
                        name="decode_hca_compress_state_block_table",
                    ),
                    "csa_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.prefill_csa_state_max_blocks),
                        torch.int32,
                        name="decode_csa_compress_state_block_table",
                    ),
                    "csa_inner_compress_state_block_table": self._shared_empty(
                        (ranks, batch, layout.prefill_csa_inner_state_max_blocks),
                        torch.int32,
                        name="decode_csa_inner_compress_state_block_table",
                    ),
                    "hca_cmp_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_hca_cmp_slot_mapping",
                    ),
                    "hca_state_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_hca_state_slot_mapping",
                    ),
                    "csa_cmp_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_cmp_slot_mapping",
                    ),
                    "csa_idx_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_idx_slot_mapping",
                    ),
                    "csa_state_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_state_slot_mapping",
                    ),
                    "csa_inner_state_slot_mapping": self._shared_empty(
                        (ranks, tokens),
                        torch.long,
                        name="decode_csa_inner_state_slot_mapping",
                    ),
                },
            )
            self._decode_buffers = buffers
        return buffers

    def _ensure_mtp_buffers(self, hidden_size: int) -> _DeepSeekV4MtpSharedBuffers | None:
        """Allocate and stage all MTP-owned shared tensors before worker fork."""
        if self._compiled.mtp_prefill is None or self._compiled.mtp_decode is None:
            return None
        if self._mtp_buffers is not None:
            return self._mtp_buffers
        self._ensure_shared_host_allocation_before_worker("mtp buffers")
        layout = self._compiled.layout
        ranks = layout.ranks
        tokens = layout.decode_tokens
        hidden = int(hidden_size)
        loaded = self.load_mtp_weights()
        weights = {
            name: self._new_shared_like(tensor, name=f"mtp_weight[{name}]")
            for name, tensor in loaded.tensors.items()
        }
        for name, tensor in loaded.tensors.items():
            self._copy_shared(weights[name], tensor, name=f"mtp_weight[{name}]")
        mtp_kv_cache = self._shared_empty(
            (ranks, layout.prefill_ori_max_blocks, layout.block_size, 1, DEEPSEEK_V4_HEAD_DIM),
            torch.bfloat16,
            name="mtp_unified_kv_cache",
        )
        self._mtp_buffers = _DeepSeekV4MtpSharedBuffers(
            weights=weights,
            prefill_hidden_in=self._shared_empty(
                (ranks, layout.prefill_seq, hidden), torch.bfloat16, name="mtp_prefill_hidden_in"
            ),
            prefill_prev_hidden_in=self._shared_empty(
                (ranks, layout.prefill_seq, layout.hc_mult, hidden),
                torch.float32,
                name="mtp_prefill_prev_hidden_in",
            ),
            prefill_input_ids=self._shared_empty(
                (ranks, layout.prefill_seq), torch.long, name="mtp_prefill_input_ids"
            ),
            prefill_position_ids=self._shared_empty(
                (ranks, layout.prefill_seq), torch.int32, name="mtp_prefill_position_ids"
            ),
            prefill_block_table=self._shared_empty(
                (ranks, layout.prefill_ori_max_blocks), torch.int32, name="mtp_prefill_block_table"
            ),
            prefill_slot_mapping=self._shared_empty(
                (ranks, layout.prefill_seq), torch.long, name="mtp_prefill_slot_mapping"
            ),
            prefill_kv_cache=mtp_kv_cache,
            decode_hidden_in=self._shared_empty(
                (ranks, tokens, hidden), torch.bfloat16, name="mtp_decode_hidden_in"
            ),
            decode_prev_hidden_in=self._shared_empty(
                (ranks, tokens, layout.hc_mult, hidden),
                torch.float32,
                name="mtp_decode_prev_hidden_in",
            ),
            decode_input_ids=self._shared_empty(
                (ranks, tokens), torch.long, name="mtp_decode_input_ids"
            ),
            decode_position_ids=self._shared_empty(
                (ranks, tokens), torch.int32, name="mtp_decode_position_ids"
            ),
            decode_slot_mapping=self._shared_empty(
                (ranks, tokens), torch.long, name="mtp_decode_slot_mapping"
            ),
            decode_swa_indices=self._shared_empty(
                (ranks, tokens, layout.sliding_window), torch.int32, name="mtp_decode_swa_indices"
            ),
            decode_swa_lens=self._shared_empty(
                (ranks, tokens), torch.int32, name="mtp_decode_swa_lens"
            ),
            decode_kv_cache=mtp_kv_cache,
            prefill_hidden_out=self._shared_empty(
                (ranks, layout.prefill_seq, hidden), torch.bfloat16, name="mtp_prefill_hidden_out"
            ),
            prefill_pre_hc_out=self._shared_empty(
                (ranks, layout.prefill_seq, layout.hc_mult, hidden),
                torch.float32,
                name="mtp_prefill_pre_hc_out",
            ),
            decode_hidden_out=self._shared_empty(
                (ranks, tokens, hidden), torch.bfloat16, name="mtp_decode_hidden_out"
            ),
            decode_pre_hc_out=self._shared_empty(
                (ranks, tokens, layout.hc_mult, hidden),
                torch.float32,
                name="mtp_decode_pre_hc_out",
            ),
        )
        self._mtp_buffers.prefill_kv_cache.zero_()
        return self._mtp_buffers

    def _stage_decode_inputs(self, inputs: DeepSeekV4PreparedDecodeInputs) -> DeepSeekV4PreparedDecodeInputs:
        buffers = self._ensure_decode_buffers(inputs.x_hc.shape[-1])
        self._copy_shared(buffers.x_hc_a, inputs.x_hc, name="decode_x_hc")
        staged_values: dict[str, torch.Tensor] = {}
        for name in _DECODE_INPUT_TENSOR_FIELDS:
            dst = buffers.tensors[name]
            self._copy_shared(dst, getattr(inputs, name), name=f"decode_{name}")
            staged_values[name] = dst
        return replace(inputs, x_hc=buffers.x_hc_a, **staged_values)

    def _ensure_prefill_fwd_buffers(self, hidden_size: int) -> _DeepSeekV4PrefillFwdSharedBuffers:
        """Allocate the layer-stacked shared buffers for the packed prefill dispatch."""
        buffers = self._prefill_fwd_buffers
        if buffers is not None:
            return buffers
        self._ensure_shared_host_allocation_before_worker("prefill_fwd buffers")
        layout = self._compiled.layout
        ranks = layout.ranks
        seq = layout.prefill_seq
        hidden = int(hidden_size)
        cache = self._require_decode_work_cache()
        rope_dim = self._compiled.freqs_cos.shape[-1] if self._compiled.freqs_cos is not None else 0
        max_seq_len = self._compiled.freqs_cos.shape[0] if self._compiled.freqs_cos is not None else 0

        def shared(shape, dtype, name):
            return self._shared_empty(shape, dtype, name=name)

        tensors: dict[str, torch.Tensor] = {
            # Prefill and decode bind the exact same persistent physical pools.
            "hca_compress_state": cache.hca_compress_state,
            "hca_compress_state_block_table": shared(
                (ranks, layout.prefill_hca_state_max_blocks), torch.int32, "prefill_fwd_hca_state_block_table"
            ),
            "csa_compress_state": cache.csa_compress_state,
            "csa_compress_state_block_table": shared(
                (ranks, layout.prefill_csa_state_max_blocks), torch.int32, "prefill_fwd_csa_state_block_table"
            ),
            "csa_inner_compress_state": cache.csa_inner_compress_state,
            "csa_inner_compress_state_block_table": shared(
                (ranks, layout.prefill_csa_inner_state_max_blocks), torch.int32, "prefill_fwd_csa_inner_state_block_table"
            ),
            "kv_cache": cache.kv_cache,
            "cmp_kv": cache.cmp_kv,
            "idx_kv_cache": cache.idx_kv_cache,
            # Per-token quant scale paired with the INT8 idx_kv_cache (ratio-4
            # indexer cache is now quant-on-write; scale is FP32, last dim 1).
            "idx_kv_scale": cache.idx_kv_scale,
            # Shared single per-rank metadata (the kernel passes each whole tensor
            # to every layer).
            "ori_block_table": shared(
                (ranks, layout.prefill_ori_max_blocks), torch.int32, "prefill_fwd_ori_block_table"
            ),
            "ori_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_ori_slot_mapping"),
            "cmp_block_table": shared((ranks, layout.prefill_cmp_max_blocks), torch.int32, "prefill_fwd_cmp_block_table"),
            "idx_block_table": shared((ranks, layout.prefill_idx_max_blocks), torch.int32, "prefill_fwd_idx_block_table"),
            "position_ids": shared((ranks, seq), torch.int32, "prefill_fwd_position_ids"),
            "hca_cmp_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_hca_cmp_slot_mapping"),
            "hca_state_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_hca_state_slot_mapping"),
            "csa_cmp_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_csa_cmp_slot_mapping"),
            "csa_idx_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_csa_idx_slot_mapping"),
            "csa_state_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_csa_state_slot_mapping"),
            "csa_inner_state_slot_mapping": shared((ranks, seq), torch.long, "prefill_fwd_csa_inner_state_slot_mapping"),
            "input_ids": shared((ranks, seq), torch.long, "prefill_fwd_input_ids"),
        }
        buffers = _DeepSeekV4PrefillFwdSharedBuffers(
            x_hc=shared((ranks, seq, layout.hc_mult, hidden), torch.float32, "prefill_fwd_x_hc"),
            freqs_cos=shared((ranks, max_seq_len, rope_dim), torch.bfloat16, "prefill_fwd_freqs_cos"),
            freqs_sin=shared((ranks, max_seq_len, rope_dim), torch.bfloat16, "prefill_fwd_freqs_sin"),
            tensors=tensors,
        )
        self._prefill_fwd_buffers = buffers
        return buffers

    def _require_prefill_fwd_buffers(self) -> _DeepSeekV4PrefillFwdSharedBuffers:
        if self._prefill_fwd_buffers is None:
            raise RuntimeError("DeepSeekV4 packed prefill shared buffers were not staged")
        return self._prefill_fwd_buffers

    def _stage_prefill_fwd_inputs(self, inputs: DeepSeekV4PreparedPrefillInputs) -> None:
        """Copy one prefill chunk's metadata/state into the packed buffers.

        The per-request metadata (slot mappings, block tables, position/input
        ids), the RoPE tables and the compressor-state block tables
        are shared single per-rank copies (the kernel slices them per layer
        internally). The compressor-state and work caches start zeroed and are
        produced by the kernel.
        """
        buffers = self._require_prefill_fwd_buffers()

        # x_hc / output collapse weights.
        self._copy_shared(buffers.x_hc, inputs.x_hc, name="prefill_fwd_x_hc")
        self._copy_shared(
            buffers.freqs_cos,
            self._static_freqs_cos_table(),
            name="prefill_fwd_freqs_cos",
        )
        self._copy_shared(
            buffers.freqs_sin,
            self._static_freqs_sin_table(),
            name="prefill_fwd_freqs_sin",
        )

        # Shared single per-rank metadata (the kernel slices it per layer).
        shared_metadata = {
            "ori_block_table": inputs.ori_block_table,
            "ori_slot_mapping": inputs.ori_slot_mapping,
            "cmp_block_table": inputs.cmp_block_table,
            "idx_block_table": inputs.idx_block_table,
            "position_ids": inputs.position_ids,
            "hca_cmp_slot_mapping": inputs.hca_cmp_slot_mapping,
            "hca_state_slot_mapping": inputs.hca_state_slot_mapping,
            "csa_cmp_slot_mapping": inputs.csa_cmp_slot_mapping,
            "csa_idx_slot_mapping": inputs.csa_idx_slot_mapping,
            "csa_state_slot_mapping": inputs.csa_state_slot_mapping,
            "csa_inner_state_slot_mapping": inputs.csa_inner_state_slot_mapping,
            "input_ids": inputs.input_ids,
            "hca_compress_state_block_table": inputs.hca_compress_state_block_table,
            "csa_compress_state_block_table": inputs.csa_compress_state_block_table,
            "csa_inner_compress_state_block_table": inputs.csa_inner_compress_state_block_table,
        }
        for name, tensor in shared_metadata.items():
            self._copy_shared(buffers.tensors[name], tensor, name=f"prefill_fwd_{name}")

        # Initialize only this request's physical pages once. The unified pools
        # use ``torch.empty`` and slots are reused, so every KV/compressor family
        # must discard state left by the previous owner without touching pages
        # that belong to other active slots.
        if inputs.request_id not in self._initialized_cache_requests:
            self._zero_unified_cache_slot(inputs.slot)
            self._initialized_cache_requests.add(inputs.request_id)

    def _zero_unified_cache_slot(self, slot: int) -> None:
        """Zero the physical-page partition assigned to one serving slot."""
        layout = self._compiled.layout
        cache = self._require_decode_work_cache()

        def zero_pages(tensor: torch.Tensor, layers: int, physical_blocks: int) -> None:
            pages = self.cache_manager.block_table(
                [slot], max_blocks=physical_blocks, physical_blocks=physical_blocks
            )[0].unique().to(torch.long)
            tensor.reshape(
                layout.ranks,
                layers,
                physical_blocks,
                *tensor.shape[2:],
            ).index_fill_(2, pages, 0)

        zero_pages(cache.kv_cache, DEEPSEEK_V4_FWD_NUM_LAYERS, layout.decode_ori_max_blocks)
        zero_pages(cache.cmp_kv, DEEPSEEK_V4_FWD_NUM_LAYERS, layout.cmp_max_blocks)
        zero_pages(cache.idx_kv_cache, DEEPSEEK_V4_CSA_NUM_LAYERS, layout.idx_max_blocks)
        zero_pages(cache.idx_kv_scale, DEEPSEEK_V4_CSA_NUM_LAYERS, layout.idx_max_blocks)
        zero_pages(cache.hca_compress_state, DEEPSEEK_V4_HCA_NUM_LAYERS, layout.hca_state_max_blocks)
        zero_pages(cache.csa_compress_state, DEEPSEEK_V4_CSA_NUM_LAYERS, layout.csa_state_max_blocks)
        zero_pages(
            cache.csa_inner_compress_state,
            DEEPSEEK_V4_CSA_NUM_LAYERS,
            layout.csa_inner_state_max_blocks,
        )
        if self._mtp_buffers is not None:
            zero_pages(self._mtp_buffers.prefill_kv_cache, 1, layout.decode_ori_max_blocks)

    def _static_freqs_cos_table(self) -> torch.Tensor:
        if self._compiled.freqs_cos is None:
            raise RuntimeError("DeepSeekV4 RoPE cosine table is not initialized")
        return self._rank_stack(self._compiled.freqs_cos)

    def _static_freqs_sin_table(self) -> torch.Tensor:
        if self._compiled.freqs_sin is None:
            raise RuntimeError("DeepSeekV4 RoPE sine table is not initialized")
        return self._rank_stack(self._compiled.freqs_sin)

    def _stage_stacked_weights(self, weights: DeepSeekV4StackedLayerWeights) -> DeepSeekV4StackedLayerWeights:
        """Copy the layer-stacked decode_fwd weights into shared buffers once."""
        buffers = self._stacked_weight_buffers
        if buffers is None:
            self._ensure_shared_host_allocation_before_worker("stacked layer weights")
            buffers = {
                name: self._new_shared_like(tensor, name=f"stacked_weight[{name}]")
                for name, tensor in weights.tensors.items()
            }
            self._stacked_weight_buffers = buffers

        missing = sorted(set(weights.tensors) - set(buffers))
        if missing:
            raise KeyError(f"DeepSeekV4 shared stacked-weight buffers are missing: {', '.join(missing)}")

        for name, tensor in weights.tensors.items():
            self._copy_shared(buffers[name], tensor, name=f"stacked_weight[{name}]")
        return DeepSeekV4StackedLayerWeights(tensors=buffers)

    def _hc_head_tensors(self) -> dict[str, torch.Tensor]:
        """Return rank-replicated hc_head weights for the decode_fwd output collapse."""
        buffers = self._hc_head_buffers
        if buffers is not None:
            return buffers
        self._ensure_shared_host_allocation_before_worker("hc_head weights")
        global_weights = self.load_packed_global_weights()
        ranks = self._compiled.layout.ranks
        # The kernel hc_head_fn is [HC_MULT, HC_DIM]; the checkpoint stores it as
        # [HC_MULT, hidden*HC_MULT] (== [HC_MULT, HC_DIM]). Scale/base are scalars
        # per HC_MULT row, rank-replicated.
        hc_head_fn = global_weights.hc_head_fn.to(torch.float32).contiguous().cpu()
        hc_head_scale = global_weights.hc_head_scale.to(torch.float32).contiguous().cpu()
        hc_head_base = global_weights.hc_head_base.to(torch.float32).contiguous().cpu()
        buffers = {
            "hc_head_fn": self._static_device_tensor(self._rank_stack(hc_head_fn)),
            "hc_head_scale": self._static_device_tensor(self._rank_stack(hc_head_scale)),
            "hc_head_base": self._static_device_tensor(self._rank_stack(hc_head_base)),
        }
        self._hc_head_buffers = buffers
        return buffers

    def _require_decode_buffers(self) -> _DeepSeekV4DecodeSharedBuffers:
        if self._decode_buffers is None:
            raise RuntimeError("DeepSeekV4 decode shared buffers were not staged")
        return self._decode_buffers

    def _require_decode_output_buffer(self, hidden_size: int) -> torch.Tensor:
        return self._ensure_decode_buffers(int(hidden_size)).x_out

    def _require_decode_logits_buffer(self, vocab_size: int) -> torch.Tensor:
        """Return a legacy shared ``[ranks, decode_tokens, vocab]`` logits buffer."""
        layout = self._compiled.layout
        logits_shape = (layout.ranks, layout.decode_tokens, int(vocab_size))
        if self._decode_logits_buffer is None:
            self._ensure_shared_host_allocation_before_worker("decode_logits")
            self._decode_logits_buffer = self._shared_empty(logits_shape, torch.float32, name="decode_logits")
        return self._decode_logits_buffer

    def _require_prefill_output_buffer(self, hidden_size: int) -> torch.Tensor:
        """Return the shared ``[ranks, prefill_seq, hidden]`` prefill hidden output."""
        layout = self._compiled.layout
        output_shape = (layout.ranks, layout.prefill_seq, int(hidden_size))
        if self._prefill_output_buffer is None:
            self._ensure_shared_host_allocation_before_worker("prefill_output")
            self._prefill_output_buffer = self._shared_empty(output_shape, torch.bfloat16, name="prefill_output")
        return self._prefill_output_buffer

    def _require_prefill_pre_hc_output_buffer(self, hidden_size: int) -> torch.Tensor:
        """Return the main-model final pre-HC rows used to seed MTP prefill."""
        layout = self._compiled.layout
        output_shape = (layout.ranks, layout.prefill_seq, layout.hc_mult, int(hidden_size))
        if self._prefill_pre_hc_output_buffer is None:
            self._ensure_shared_host_allocation_before_worker("prefill_pre_hc_output")
            self._prefill_pre_hc_output_buffer = self._shared_empty(
                output_shape,
                torch.float32,
                name="prefill_pre_hc_output",
            )
        return self._prefill_pre_hc_output_buffer

    def _static_final_norm_weight_tensor(self) -> torch.Tensor:
        """Return the worker-resident per-rank final RMSNorm weight ``[ranks, D]``.

        Reuses the same ``final_norm_weight`` already loaded for the host-side
        ``_final_norm`` collapse, rank-replicated and cast to bf16 for the kernel.
        """
        if self._static_final_norm_weight is None:
            global_weights = self.load_packed_global_weights()
            self._ensure_shared_host_allocation_before_worker("final_norm_w")
            final_norm_w = global_weights.final_norm_weight.to(torch.bfloat16).contiguous().cpu()
            self._static_final_norm_weight = self._static_device_tensor(self._rank_stack(final_norm_w))
        return self._static_final_norm_weight

    def _static_freqs_cos_tensor(self) -> torch.Tensor:
        if self._static_freqs_cos is None:
            if self._compiled.freqs_cos is None:
                raise RuntimeError("DeepSeekV4 RoPE cosine table is not initialized")
            self._ensure_shared_host_allocation_before_worker("freqs_cos")
            self._static_freqs_cos = self._static_device_tensor(self._rank_stack(self._compiled.freqs_cos))
        return self._static_freqs_cos

    def _static_freqs_sin_tensor(self) -> torch.Tensor:
        if self._static_freqs_sin is None:
            if self._compiled.freqs_sin is None:
                raise RuntimeError("DeepSeekV4 RoPE sine table is not initialized")
            self._ensure_shared_host_allocation_before_worker("freqs_sin")
            self._static_freqs_sin = self._static_device_tensor(self._rank_stack(self._compiled.freqs_sin))
        return self._static_freqs_sin

    def _logits_for_hidden(
        self,
        x_hc: torch.Tensor,
        *,
        active_rows: Sequence[int],
        label: str = "unknown",
    ) -> torch.Tensor:
        global_weights = self.load_packed_global_weights()
        if x_hc.ndim == 3:
            # Decode output is already collapsed and final-normalized by
            # ``l3_decode_fwd``; host LM-head consumes it directly.
            hidden = x_hc
        else:
            hidden = self._final_hidden(x_hc)
        rows = tuple(int(row) for row in active_rows)
        if not rows:
            raise ValueError("DeepSeekV4 LM-head requires at least one active row")
        if min(rows) < 0 or max(rows) >= hidden.shape[1]:
            raise ValueError(
                f"DeepSeekV4 LM-head active rows {rows} exceed hidden rows={hidden.shape[1]}"
            )
        row_list = list(rows)
        if self._debug_tensor_stats_enabled():
            print(f"DSV4_DEBUG lm_head.label={label} active_rows={rows}", flush=True)
            if x_hc.ndim == 4:
                self._debug_tensor_stats("lm_head.x_hc.active", x_hc[:, row_list, :, :])
            self._debug_tensor_stats("lm_head.hidden.active", hidden[:, row_list, :])

        layout = global_weights.lm_head_layout
        if global_weights.lm_head_weight.shape[0] != layout.ranks:
            raise ValueError(
                "DeepSeekV4 packed LM-head rank count mismatch: "
                f"weight ranks={global_weights.lm_head_weight.shape[0]} layout ranks={layout.ranks}"
            )
        if global_weights.lm_head_weight.shape[1] < layout.vocab_per_rank:
            raise ValueError(
                "DeepSeekV4 packed LM-head shard is smaller than the real vocab shard: "
                f"shape={tuple(global_weights.lm_head_weight.shape)} vocab_per_rank={layout.vocab_per_rank}"
            )

        selected = hidden[0, row_list, :].detach().cpu().to(torch.float32).contiguous()
        logits_parts = []
        for rank in range(layout.ranks):
            shard = global_weights.lm_head_weight[rank, : layout.vocab_per_rank, :]
            shard = shard.detach().cpu().to(torch.float32).contiguous()
            logits_parts.append(torch.matmul(selected, shard.t()))
        logits = torch.cat(logits_parts, dim=-1)
        if logits.shape[-1] != layout.vocab_size:
            logits = logits[:, : layout.vocab_size].contiguous()
        else:
            logits = logits.contiguous()
        self._debug_tensor_stats("lm_head.logits.returned", logits)
        return logits

    @staticmethod
    def _debug_tensor_stats_enabled() -> bool:
        return os.getenv("PYPTO_DSV4_LOGIT_DEBUG") == "1"

    @staticmethod
    def _debug_tensor_stats(name: str, tensor: torch.Tensor, *, per_rank: bool = False) -> None:
        if not DeepSeekV4ModelRunner._debug_tensor_stats_enabled():
            return
        data = tensor.detach().cpu().to(torch.float32)
        finite = torch.isfinite(data)
        finite_count = int(finite.sum().item())
        total = data.numel()
        nan_count = int(torch.isnan(data).sum().item())
        pos_inf_count = int(torch.isposinf(data).sum().item())
        neg_inf_count = int(torch.isneginf(data).sum().item())
        if finite_count:
            finite_values = data[finite]
            min_value = float(finite_values.min().item())
            max_value = float(finite_values.max().item())
            absmax_value = float(finite_values.abs().max().item())
        else:
            min_value = float("nan")
            max_value = float("nan")
            absmax_value = float("nan")
        print(
            "DSV4_DEBUG "
            f"{name} shape={tuple(tensor.shape)} dtype={tensor.dtype} "
            f"finite={finite_count}/{total} nan={nan_count} "
            f"+inf={pos_inf_count} -inf={neg_inf_count} "
            f"min={min_value:.6g} max={max_value:.6g} absmax={absmax_value:.6g}",
            flush=True,
        )
        if per_rank and data.ndim >= 1:
            rank_view = data.reshape(data.shape[0], -1)
            rank_finite = torch.isfinite(rank_view)
            rank_counts = (rank_view.shape[1] - rank_finite.sum(dim=1)).tolist()
            print(f"DSV4_DEBUG {name} nonfinite_by_rank={rank_counts}", flush=True)

    @staticmethod
    def _tensor_is_finite(tensor: torch.Tensor) -> bool:
        return bool(torch.isfinite(tensor.detach().cpu().to(torch.float32)).all().item())

    def _final_hidden(self, x_hc: torch.Tensor) -> torch.Tensor:
        """Collapse a ``[ranks, T, HC_MULT, D]`` HC stack and apply the final norm."""
        weights = self.load_packed_global_weights()
        x_hc = x_hc.to(torch.bfloat16).cpu()
        x_float = x_hc.float()
        flat = x_float.flatten(2)
        rms = torch.sqrt(flat.double().square().mean(dim=-1, keepdim=True) + DEEPSEEK_V4_RMS_NORM_EPS)
        normed_flat = flat / rms.to(torch.float32)
        mixes = torch.matmul(normed_flat, weights.hc_head_fn.t())
        pre = torch.sigmoid(mixes * weights.hc_head_scale + weights.hc_head_base) + DEEPSEEK_V4_HC_EPS
        collapsed = torch.sum(pre.unsqueeze(-1).double() * x_float.double(), dim=2)
        return self._final_norm(collapsed)

    def _final_norm(self, collapsed: torch.Tensor) -> torch.Tensor:
        """Apply the final RMS norm to an already-collapsed ``[ranks, T, D]`` hidden.

        The packed ``l3_decode_fwd`` kernel collapses HC_MULT in-kernel via
        ``hc_head`` and returns the collapsed (pre-final-norm) hidden, so decode
        only needs the model's final RMS norm before the LM head.
        """
        collapsed = collapsed.cpu().double()
        weights = self.load_packed_global_weights()
        norm_inv = torch.rsqrt(collapsed.square().mean(dim=-1, keepdim=True) + DEEPSEEK_V4_RMS_NORM_EPS)
        normed = collapsed * norm_inv * weights.final_norm_weight.double()
        return normed.to(torch.float32).to(torch.bfloat16).contiguous()

    def _scope_stats_run_config(self) -> Any:
        """Optional per-dispatch RunConfig that captures device scope stats.

        Enabled with ``PYPTO_DSV4_SCOPE_STATS=1`` to dump per-scope
        heap / task_window / tensormap peaks under ``<dir>/dfx_outputs/``.
        """
        if os.getenv("PYPTO_DSV4_SCOPE_STATS") != "1":
            return None
        from pypto.runtime import RunConfig  # noqa: PLC0415

        out_dir = os.getenv("PYPTO_DSV4_SCOPE_STATS_DIR", "/data/liuxu/pypto-serving/dsv4_scope_stats")
        return RunConfig(
            platform=self._compiled.platform,
            device_id=self._compiled.device_id,
            enable_scope_stats=True,
            save_kernels=True,
            save_kernels_dir=out_dir,
        )

    def _run_l3(self, callable_spec: DeepSeekV4L3Callable, *args: Any) -> Any:
        if self._l3_worker is None:
            self._assert_l3_args_shared_before_worker(callable_spec, args)
        worker = self._shared_l3_worker()
        run_config = self._scope_stats_run_config()
        uploaded: list[DeviceTensor] = []
        try:
            l3_args = tuple(self._coerce_l3_arg(worker, arg, uploaded) for arg in args)
            if run_config is not None:
                return worker.run(callable_spec.compiled, *l3_args, config=run_config)
            return worker.run(callable_spec.compiled, *l3_args)
        finally:
            for tensor in uploaded:
                worker.free_tensor(tensor)

    @staticmethod
    def _share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        if not tensor.is_shared():
            tensor = tensor.share_memory_()
        return tensor

    @staticmethod
    def _shared_empty(shape: Sequence[int], dtype: torch.dtype, *, name: str) -> torch.Tensor:
        del name
        return torch.empty(tuple(int(dim) for dim in shape), dtype=dtype).share_memory_()

    @staticmethod
    def _new_shared_like(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU tensor")
        return torch.empty_like(tensor.contiguous(), memory_format=torch.contiguous_format).share_memory_()

    @staticmethod
    def _copy_shared(dst: torch.Tensor, src: torch.Tensor, *, name: str) -> None:
        if src.device.type != "cpu":
            src = src.cpu()
        if not src.is_contiguous():
            src = src.contiguous()
        if tuple(dst.shape) != tuple(src.shape) or dst.dtype != src.dtype:
            raise ValueError(
                f"{name} shared buffer shape/dtype mismatch: "
                f"buffer shape={tuple(dst.shape)} dtype={dst.dtype}, "
                f"source shape={tuple(src.shape)} dtype={src.dtype}"
            )
        dst.copy_(src)

    @staticmethod
    def _int32_scalar(value: int) -> int:
        return int(value)

    def _ensure_shared_host_allocation_before_worker(self, name: str) -> None:
        if self._l3_worker is not None:
            raise RuntimeError(
                f"DeepSeekV4 shared host buffer '{name}' must be allocated before the L3 worker starts"
            )

    def _assert_l3_args_shared_before_worker(
        self,
        callable_spec: DeepSeekV4L3Callable,
        args: Sequence[Any],
    ) -> None:
        for index, arg in enumerate(args):
            self._assert_l3_arg_shared(arg, name=f"{callable_spec.name}[{index}]")

    def _assert_l3_arg_shared(self, arg: Any, *, name: str) -> None:
        if isinstance(arg, (_StaticDeviceTensor, _TransientDeviceTensor)):
            self._assert_l3_arg_shared(arg.tensor, name=f"{name}.tensor")
            return
        if isinstance(arg, torch.Tensor) and arg.device.type == "cpu" and not arg.is_shared():
            raise TypeError(
                "DeepSeekV4 L3 dispatch requires shared-memory CPU tensors allocated before "
                f"the L3 worker starts; got {name} shape={tuple(arg.shape)} dtype={arg.dtype}"
            )
        if isinstance(arg, Sequence) and not isinstance(arg, (str, bytes, bytearray)):
            for index, item in enumerate(arg):
                self._assert_l3_arg_shared(item, name=f"{name}[{index}]")
            return
        if isinstance(arg, dict):
            for key, item in arg.items():
                self._assert_l3_arg_shared(item, name=f"{name}[{key!r}]")

    def _coerce_l3_arg(self, worker: Any, arg: Any, uploaded: list[DeviceTensor]) -> Any:
        if isinstance(arg, _StaticDeviceTensor):
            self._assert_l3_arg_shared(arg, name="static")
            tensor = arg.tensor
            key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
            cached = self._l3_static_tensors.get(key)
            if cached is None:
                cached = worker.alloc_stacked_tensor(tensor)
                self._l3_static_tensors[key] = cached
            if arg.cache_state:
                self._l3_cache_tensor_keys.add(key)
            return cached
        if isinstance(arg, _TransientDeviceTensor):
            tensor = arg.tensor
            self._assert_l3_arg_shared(arg, name="transient")
            dev = worker.alloc_tensor(tensor.shape, tensor.dtype, init=tensor)
            uploaded.append(dev)
            return dev
        if isinstance(arg, torch.Tensor) and arg.device.type == "cpu" and not arg.is_shared():
            raise TypeError(
                "DeepSeekV4 L3 dispatch requires shared-memory CPU tensors allocated before "
                f"the worker starts; got non-shared tensor shape={tuple(arg.shape)} dtype={arg.dtype}"
            )
        return arg

    @staticmethod
    def _mark_resident_args(
        values: dict[str, Any],
        policy: dict[str, bool],
    ) -> dict[str, Any]:
        """Mark policy-selected arguments for lazy one-time Device upload."""
        missing = [name for name in policy if name not in values]
        if missing:
            raise KeyError(f"DeepSeekV4 resident argument policy is missing values: {missing}")
        for name, cache_state in policy.items():
            tensor = values[name]
            if isinstance(tensor, StackedDeviceTensor):
                continue
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"DeepSeekV4 resident argument {name!r} must be a torch.Tensor or "
                    f"StackedDeviceTensor, got {type(tensor).__name__}"
                )
            if tensor.device.type != "cpu" or not tensor.is_contiguous() or not tensor.is_shared():
                raise ValueError(
                    f"DeepSeekV4 resident argument {name!r} must be a contiguous shared-memory "
                    "CPU tensor"
                )
            values[name] = _StaticDeviceTensor(tensor=tensor, cache_state=cache_state)
        return values

    @staticmethod
    def _upload_weight_group(
        worker: Any,
        host_weights: dict[str, torch.Tensor],
    ) -> dict[str, StackedDeviceTensor]:
        """Upload a rank-stacked weight group, rolling back partial allocation."""
        device_weights: dict[str, StackedDeviceTensor] = {}
        try:
            for name, tensor in host_weights.items():
                device_weights[name] = worker.alloc_stacked_tensor(tensor)
        except Exception:
            for tensor in device_weights.values():
                worker.free_stacked_tensor(tensor)
            raise
        return device_weights

    def _materialize_resident_weights(self) -> None:
        """Upload main/MTP weights once and release their shared Host backing."""
        worker = self._shared_l3_worker()
        if self._stacked_device_weights is None:
            host_weights = self._stacked_weight_buffers
            if not host_weights:
                raise RuntimeError("DeepSeekV4 stacked Host weights are not staged")
            host_bytes = sum(tensor.numel() * tensor.element_size() for tensor in host_weights.values())
            self._stacked_device_weights = self._upload_weight_group(worker, host_weights)
            self._stacked_weight_buffers = None
            logger.info(
                "DeepSeekV4 resident main weights uploaded; released_host_bytes=%d",
                host_bytes,
            )

        buffers = self._mtp_buffers
        if buffers is not None and self._mtp_device_weights is None:
            if not buffers.weights:
                raise RuntimeError("DeepSeekV4 MTP Host weights are not staged")
            host_bytes = sum(tensor.numel() * tensor.element_size() for tensor in buffers.weights.values())
            self._mtp_device_weights = self._upload_weight_group(worker, buffers.weights)
            buffers.weights.clear()
            logger.info(
                "DeepSeekV4 resident MTP weights uploaded; released_host_bytes=%d",
                host_bytes,
            )

    def _invalidate_resident_cache_tensors(self) -> None:
        """Free resident KV/compressor state so the next request starts clean."""
        worker = self._l3_worker
        if worker is None:
            self._l3_cache_tensor_keys.clear()
            return
        for key in tuple(self._l3_cache_tensor_keys):
            tensor = self._l3_static_tensors.pop(key, None)
            if tensor is not None:
                worker.free_stacked_tensor(tensor)
        self._l3_cache_tensor_keys.clear()

    def _shared_l3_worker(self) -> Any:
        worker = self._l3_worker
        if worker is None:
            self._assert_l3_shared_buffers_preallocated()
            compiled_callables = self._compiled.l3_callables()
            if not compiled_callables:
                raise RuntimeError("DeepSeekV4 L3 callables are not compiled")
            from pypto.runtime import DistributedWorker  # noqa: PLC0415

            worker = DistributedWorker([callable_spec.compiled for callable_spec in compiled_callables])
            self._l3_worker = worker
        return worker

    def _ensure_decode_work_cache(self) -> DeepSeekV4LayerCache:
        cache = self._decode_work_cache
        if cache is not None:
            return cache
        self._ensure_shared_host_allocation_before_worker("decode work cache")
        layout = self._compiled.layout
        fwd_layers = DEEPSEEK_V4_FWD_NUM_LAYERS
        csa_layers = DEEPSEEK_V4_CSA_NUM_LAYERS
        hca_layers = DEEPSEEK_V4_HCA_NUM_LAYERS
        cache = DeepSeekV4LayerCache(
            kv_cache=self._shared_empty(
                (
                    layout.ranks,
                    fwd_layers * layout.decode_ori_max_blocks,
                    layout.block_size,
                    1,
                    DEEPSEEK_V4_HEAD_DIM,
                ),
                torch.bfloat16,
                name="decode_work_kv_cache",
            ),
            cmp_kv=self._shared_empty(
                (
                    layout.ranks,
                    fwd_layers * layout.cmp_max_blocks,
                    layout.block_size,
                    1,
                    DEEPSEEK_V4_HEAD_DIM,
                ),
                torch.bfloat16,
                name="decode_work_cmp_kv",
            ),
            idx_kv_cache=self._shared_empty(
                (
                    layout.ranks,
                    csa_layers * layout.idx_max_blocks,
                    layout.block_size,
                    1,
                    DEEPSEEK_V4_IDX_HEAD_DIM,
                ),
                torch.int8,
                name="decode_work_idx_kv_cache",
            ),
            idx_kv_scale=self._shared_empty(
                (
                    layout.ranks,
                    csa_layers * layout.idx_max_blocks,
                    layout.block_size,
                    1,
                    1,
                ),
                torch.float32,
                name="decode_work_idx_kv_scale",
            ),
            hca_compress_state=self._shared_empty(
                (
                    layout.ranks,
                    hca_layers * layout.hca_state_max_blocks,
                    layout.c128_state_block_size,
                    DEEPSEEK_V4_HCA_STATE_DIM,
                ),
                torch.float32,
                name="decode_work_hca_compress_state",
            ),
            csa_compress_state=self._shared_empty(
                (
                    layout.ranks,
                    csa_layers * layout.csa_state_max_blocks,
                    layout.c4_state_block_size,
                    DEEPSEEK_V4_CSA_STATE_DIM,
                ),
                torch.float32,
                name="decode_work_csa_compress_state",
            ),
            csa_inner_compress_state=self._shared_empty(
                (
                    layout.ranks,
                    csa_layers * layout.csa_inner_state_max_blocks,
                    layout.c4_state_block_size,
                    DEEPSEEK_V4_CSA_INNER_STATE_DIM,
                ),
                torch.float32,
                name="decode_work_csa_inner_compress_state",
            ),
        )
        self._decode_work_cache = cache
        return cache

    def _require_decode_work_cache(self) -> DeepSeekV4LayerCache:
        if self._decode_work_cache is None:
            raise RuntimeError("DeepSeekV4 decode work cache was not allocated before the L3 worker started")
        return self._decode_work_cache

    @staticmethod
    def _static_device_tensor(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type != "cpu":
            raise ValueError("worker-resident tensor must be on CPU")
        if not tensor.is_contiguous():
            raise ValueError("worker-resident tensor must be contiguous")
        return DeepSeekV4ModelRunner._share_cpu_tensor(tensor)

    def _reset_l3_worker(self) -> None:
        worker = self._l3_worker
        if worker is None:
            return
        try:
            worker.close()
        finally:
            self._l3_worker = None
            self._l3_static_tensors.clear()
            self._l3_cache_tensor_keys.clear()

    def close(self) -> None:
        worker = self._l3_worker
        try:
            if worker is not None:
                worker.close()
        finally:
            self._l3_worker = None
            self._decode_work_cache = None
            self._stacked_weight_buffers = None
            self._stacked_device_weights = None
            self._mtp_device_weights = None
            self._mtp_buffers = None
            self._global_weights = None
            self._initialized_cache_requests.clear()
            self._l3_static_tensors.clear()
            self._l3_cache_tensor_keys.clear()

    def _require_input_builder(self) -> DeepSeekV4InputBuilder:
        if self.input_builder is None:
            raise RuntimeError("DeepSeekV4 input builder is not initialized")
        return self.input_builder

    def _rank_stack(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(0).expand(self._compiled.layout.ranks, *tensor.shape).contiguous()

    def _prefill_kernel_tokens(self, actual_tokens: int) -> int:
        if actual_tokens <= 0:
            raise ValueError("actual_tokens must be positive")
        return self._compiled.layout.prefill_seq

    @staticmethod
    def _prefill_kernel_positions(
        positions: Sequence[int],
        *,
        kernel_tokens: int,
        max_seq_len: int,
    ) -> list[int]:
        if len(positions) <= 0:
            raise ValueError("positions must not be empty")
        if kernel_tokens < len(positions):
            raise ValueError("kernel_tokens must cover all active positions")
        start = int(positions[0])
        kernel_positions = list(range(start, start + kernel_tokens))
        if kernel_positions[-1] >= max_seq_len:
            raise ValueError(
                f"prefill static kernel position {kernel_positions[-1]} exceeds max_seq_len={max_seq_len}"
            )
        return kernel_positions

    def _prefill_kernel_slots(self, slot: int, *, actual_tokens: int, kernel_tokens: int) -> list[int]:
        if actual_tokens <= 0:
            raise ValueError("actual_tokens must be positive")
        if kernel_tokens < actual_tokens:
            raise ValueError("kernel_tokens must cover all active tokens")
        slot = int(slot)
        scratch_slot = slot
        if kernel_tokens > actual_tokens and self._compiled.layout.decode_batch > 1:
            scratch_slot = (slot + self._compiled.layout.decode_batch - 1) % self._compiled.layout.decode_batch
        return [slot] * actual_tokens + [scratch_slot] * (kernel_tokens - actual_tokens)

    def _prefill_ori_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[int],
    ) -> torch.Tensor:
        layout = self._compiled.layout
        if len(slots) != len(positions):
            raise ValueError("prefill slots and positions must have the same length")
        mapping = torch.empty((len(positions),), dtype=torch.long)
        capacity = layout.prefill_ori_max_blocks * layout.block_size
        tables = self.cache_manager.block_table(
            slots,
            max_blocks=layout.prefill_ori_max_blocks,
            physical_blocks=layout.decode_ori_max_blocks,
        )
        for row, position in enumerate(positions):
            position = int(position)
            if position < 0 or position >= capacity:
                raise ValueError(
                    f"prefill ori position {position} exceeds cache capacity {capacity}"
                )
            logical_block, intra = divmod(position, layout.block_size)
            mapping[row] = int(tables[row, logical_block]) * layout.block_size + intra
        return mapping

    def _prefill_compressed_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[int],
        *,
        max_blocks: int,
        compress_ratio: int,
    ) -> torch.Tensor:
        if len(slots) != len(positions):
            raise ValueError("prefill slots and positions must have the same length")
        if compress_ratio <= 0:
            raise ValueError("compress_ratio must be positive")
        capacity = int(max_blocks) * self._compiled.layout.block_size
        tables = self.cache_manager.block_table(slots, max_blocks=max_blocks)
        mapping = torch.full((len(positions),), -1, dtype=torch.long)
        for row, (slot, position) in enumerate(zip(slots, positions, strict=True)):
            position = int(position)
            if (position + 1) % compress_ratio != 0:
                continue
            logical = position // compress_ratio
            if logical >= capacity:
                raise ValueError(
                    f"position {position} maps to compressed row {logical}, but capacity is {capacity}"
                )
            logical_block, intra = divmod(logical, self._compiled.layout.block_size)
            mapping[row] = int(tables[row, logical_block]) * self._compiled.layout.block_size + intra
        return mapping

    def _prefill_state_slot_mapping(
        self,
        slots: Sequence[int],
        positions: Sequence[int],
        *,
        max_blocks: int,
        physical_blocks: int,
        state_block_size: int,
    ) -> torch.Tensor:
        capacity = int(max_blocks) * int(state_block_size)
        mapping = torch.empty((len(positions),), dtype=torch.long)
        if len(slots) != len(positions):
            raise ValueError("prefill slots and positions must have the same length")
        tables = self.cache_manager.block_table(
            slots,
            max_blocks=max_blocks,
            physical_blocks=physical_blocks,
        )
        for row, position in enumerate(positions):
            position = int(position)
            if position >= capacity:
                raise ValueError(
                    f"position {position} exceeds compressor-state capacity {capacity} "
                    f"(max_blocks={max_blocks}, state_block_size={state_block_size})"
                )
            logical_block, intra = divmod(position, state_block_size)
            mapping[row] = int(tables[row, logical_block]) * state_block_size + intra
        return mapping

    @staticmethod
    def _padded_rows(values: torch.Tensor, length: int) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError(f"values must be rank-2, got shape={tuple(values.shape)}")
        if values.shape[0] <= 0:
            raise ValueError("values must not be empty")
        if values.shape[0] > length:
            raise ValueError(f"values rows {values.shape[0]} exceed padded length {length}")
        out = torch.empty((length, values.shape[1]), dtype=values.dtype, device=values.device)
        out[: values.shape[0]].copy_(values)
        if values.shape[0] < length:
            pad_rows = torch.arange(values.shape[0], length, device=values.device) % values.shape[0]
            out[values.shape[0] :].copy_(values.index_select(0, pad_rows))
        return out

    @staticmethod
    def _padded_vector(values: torch.Tensor, length: int, *, dtype: torch.dtype) -> torch.Tensor:
        if values.numel() <= 0:
            raise ValueError("values must not be empty")
        if values.numel() > length:
            raise ValueError(f"values length {values.numel()} exceeds padded length {length}")
        out = torch.empty((length,), dtype=dtype)
        out[: values.numel()] = values.to(dtype=dtype)
        if values.numel() < length:
            pad_rows = torch.arange(values.numel(), length) % values.numel()
            out[values.numel() :] = values.to(dtype=dtype).index_select(0, pad_rows)
        return out

    @staticmethod
    def _prefill_position_ids(positions: Sequence[int], length: int) -> torch.Tensor:
        if len(positions) <= 0:
            raise ValueError("positions must not be empty")
        if len(positions) > length:
            raise ValueError(f"positions length {len(positions)} exceeds padded length {length}")
        out = torch.arange(length, dtype=torch.int32)
        out[: len(positions)] = torch.tensor(tuple(int(pos) for pos in positions), dtype=torch.int32)
        return out

    @staticmethod
    def _pad_prefill_mapping(mapping: torch.Tensor, length: int) -> torch.Tensor:
        if mapping.ndim != 1:
            raise ValueError(f"prefill mapping must be rank-1, got shape={tuple(mapping.shape)}")
        if mapping.numel() > length:
            raise ValueError(f"prefill mapping length {mapping.numel()} exceeds padded length {length}")
        out = torch.full((length,), -1, dtype=mapping.dtype)
        out[: mapping.numel()].copy_(mapping.to(dtype=mapping.dtype))
        return out

    @staticmethod
    def _prefill_actual_tokens(batch: PrefillBatch) -> int:
        if batch.positions is not None:
            valid = batch.positions[0].detach().cpu()
            valid = valid[valid >= 0]
            if valid.numel() <= 0:
                raise ValueError("prefill positions must include at least one token")
            return int(valid.numel())
        seq_len = int(batch.seq_lens[0].item())
        if seq_len <= 0:
            raise ValueError("prefill seq_len must be positive")
        return seq_len

    @staticmethod
    def _prefill_positions(batch: PrefillBatch, actual_tokens: int) -> list[int]:
        if batch.positions is None:
            positions = list(range(actual_tokens))
        else:
            raw = batch.positions[0, :actual_tokens].detach().cpu().to(torch.long)
            positions = [int(pos) for pos in raw.tolist()]
        if any(pos < 0 for pos in positions):
            raise ValueError("prefill positions must be non-negative")
        expected = list(range(positions[0], positions[0] + actual_tokens))
        if positions != expected:
            raise ValueError(
                "prefill positions must form one contiguous chunk: "
                f"positions={positions[:8]}{'...' if len(positions) > 8 else ''}"
            )
        return positions

    def _decode_positions(self, batch: DecodeBatch, actual_batch: int) -> tuple[tuple[int, ...], ...]:
        decode_seq = self._compiled.layout.decode_seq
        positions = []
        for row in range(actual_batch):
            seq_len = int(batch.seq_lens[row].item())
            if seq_len < decode_seq:
                raise ValueError(
                    f"decode seq_lens must be >= decode_seq ({decode_seq}), got {seq_len}"
                )
            # MTP feeds ``decode_seq`` real trailing tokens ending at the last real
            # position ``seq_len-1`` (so positions are ``seq_len-decode_seq .. seq_len-1``).
            first_position = seq_len - decode_seq
            positions.append(tuple(first_position + offset for offset in range(decode_seq)))
        return tuple(positions)

    def _decode_token_rows(
        self,
        token_ids: torch.Tensor,
        actual_batch: int,
        *,
        vocab_size: int,
        prev_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layout = self._compiled.layout
        if token_ids.ndim == 1:
            active = token_ids[:actual_batch].reshape(actual_batch, 1)
        else:
            active = token_ids[:actual_batch, :1]
        prev_active = None
        if prev_token_ids is not None:
            prev_active = prev_token_ids[:actual_batch].reshape(actual_batch, 1)
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        rows = torch.empty(layout.decode_tokens, dtype=torch.long).reshape(
            layout.decode_batch,
            layout.decode_seq,
        )
        if prev_active is not None:
            rows.copy_(prev_active[0, 0].expand(layout.decode_batch, layout.decode_seq))
            rows[:, layout.decode_seq - 1].copy_(active[0, 0])
        else:
            rows.copy_(active[0, 0].expand(layout.decode_batch, layout.decode_seq))
        for row in range(actual_batch):
            if prev_active is not None:
                # Earlier slots use prev token; final slot uses last token.
                rows[row].copy_(prev_active[row, 0].expand(layout.decode_seq))
                rows[row, layout.decode_seq - 1].copy_(active[row, 0])
            else:
                rows[row].copy_(active[row, 0].expand(layout.decode_seq))
        return rows.reshape(layout.decode_tokens)

    def _decode_kernel_slots(self, active_slots: Sequence[int]) -> tuple[int, ...]:
        """Route padded fixed decode rows into scratch cache slots."""
        layout = self._compiled.layout
        slots = [int(slot) for slot in active_slots]
        if not slots:
            raise ValueError("decode must include at least one active slot")
        if len(set(slots)) != len(slots):
            raise ValueError(f"decode slots must be unique, got {slots}")
        if len(slots) > layout.decode_batch:
            raise ValueError(f"decode slots exceed kernel batch {layout.decode_batch}: {slots}")
        active_set = set(slots)
        for scratch_slot in range(layout.decode_batch):
            if len(slots) >= layout.decode_batch:
                break
            if scratch_slot not in active_set:
                slots.append(scratch_slot)
        if len(slots) != layout.decode_batch:
            raise RuntimeError(
                f"DeepSeekV4 decode needs {layout.decode_batch} kernel slots, built {len(slots)}"
            )
        return tuple(slots)

    def _decode_kv_seq_lens(self, seq_lens: torch.Tensor, actual_batch: int) -> torch.Tensor:
        layout = self._compiled.layout
        # The last written KV position is ``seq_len-1``, so the valid KV history
        # is exactly ``seq_len`` entries. (yangyaodong's "seq_len+1" was relative
        # to a seq_len = prompt length, which does not count the prefill token;
        # our seq_len already does.)
        active = seq_lens[:actual_batch].detach().cpu().to(torch.int32)
        return DeepSeekV4CacheManager.replicate_first_row(
            active.reshape(actual_batch, 1),
            actual_rows=actual_batch,
            kernel_rows=layout.decode_batch,
        ).reshape(layout.decode_batch)
