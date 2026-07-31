# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "pypto-lib" / "models" / "qwen3" / "14b"
QWEN_SERVING = ROOT / "pypto_serving" / "model" / "qwen"
REAL_VOCAB = 151936
PADDED_VOCAB = 152064

pytestmark = pytest.mark.skipif(
    not QWEN.is_dir(),
    reason="pypto-lib submodule is not checked out",
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_device_sampling_is_limited_by_runtime_vocab_size() -> None:
    dispatch = _source(QWEN_SERVING / "qwen3_l3_dispatch.py")
    executor = _source(QWEN_SERVING / "npu_executor.py")
    runner = _source(QWEN_SERVING / "npu_runner.py")
    constants = _source(QWEN / "constants.py")
    greedy = _source(QWEN / "greedy_sample.py")
    topk = _source(QWEN / "topk_select.py")

    assert "valid_vocab_size" not in dispatch
    assert "real_vocab=151936" in constants
    assert "REAL_VOCAB" in executor
    assert "lm_head_weight[:1].expand" in executor
    assert "valid_vocab_size" not in runner
    assert "valid_vocab_size" not in greedy
    assert "REAL_NUM_FULL_VOCAB_CHUNKS" in greedy
    assert "REAL_VOCAB_TAIL" in greedy
    assert "token_id >= pl.cast(REAL_VOCAB" in greedy
    assert "REAL_NUM_FULL_VOCAB_CHUNKS" in topk
    assert "REAL_VOCAB_TAIL" in topk
    assert "REAL_VOCAB = M.real_vocab" in topk
    assert ":REAL_VOCAB" in topk


def test_device_topk_uses_exact_grouped_selection() -> None:
    topk = _source(QWEN / "topk_select.py")
    dispatch = _source(QWEN_SERVING / "qwen3_l3_dispatch.py")
    executor = _source(QWEN_SERVING / "npu_executor.py")
    runner = _source(QWEN_SERVING / "npu_runner.py")

    assert "TOPK = 32" in topk
    assert "CHUNK_TOPK" not in topk
    assert "TOPK_GROUP_WIDTH = 2048" in topk
    assert "TOPK_NUM_GROUPS * TOPK <= TOPK_CANDIDATE_PAD" in topk
    assert "def _topk_group_pairs(" in topk
    assert "for g in pl.range(TOPK_NUM_FULL_GROUPS):" in topk
    assert "group_pairs = _topk_group_pairs(logits, b, g)" in topk
    assert "pairs = pl.mrgsort(pairs, block_len=1024)" in topk
    assert "pl.set_validshape(tail_scores_raw, 1, TOPK_GROUP_TAIL)" in topk
    assert "half0_pairs = candidate_sorted[:, 0 : 2 * TOPK]" in topk
    assert "half1_pairs = candidate_sorted[" in topk
    assert "candidate_pairs = pl.mrgsort(half0_pairs, half1_pairs)" in topk
    assert "spread_ids = torch.arange(TOPK" in topk
    assert "vals, idx = torch.topk(logits, TOPK" in topk
    assert "output_dtype=pl.INT32" in topk
    assert "qwen3_topk_select_host" in dispatch
    assert "compile_topk_select" in executor
    assert "_device_topk_outputs(" in runner
    assert "sampling_control" in topk


def test_device_greedy_tie_break_matches_host_argmax() -> None:
    greedy = _source(QWEN / "greedy_sample.py")
    topk = _source(QWEN / "topk_select.py")
    decode_path = QWEN / "decode_layer.py"
    if not decode_path.is_file():
        decode_path = QWEN / "decode_fwd.py"
    decode = _source(decode_path)

    for source in (greedy, decode):
        assert "local_token = pl.cast(0, pl.INT32)" in source
        assert "scan_c = (" in source
        assert "scan_t = (" in source
        assert "local_token = pl.cast(scan_t, pl.INT32)" in source
        assert "if val == best_val:" in source

    assert 'name_hint="greedy_select"' in topk
    assert "scan_c = (REAL_NUM_VOCAB_CHUNKS - 1) - c" in topk
    assert "scan_t = (VOCAB_CHUNK - 1) - t" in topk
    assert "local_token = pl.cast(scan_t, pl.INT32)" in topk


def test_prefill_keeps_sampling_in_standalone_device_kernel() -> None:
    prefill = _source(QWEN / "prefill_fwd.py")
    runner = _source(QWEN_SERVING / "npu_runner.py")

    assert "_greedy_sample_inline" not in prefill
    assert "_token_embed_inline" not in prefill
    assert "compiled.greedy_sample" not in runner
    assert "compiled.topk_select" in runner
    assert "_maybe_run_greedy_sample(" in runner
    assert "selection_k=1" in runner


def _device_greedy_argmax_with_clamp(logits):
    best_val = max(logits)
    token_id = 0
    for idx in range(len(logits) - 1, -1, -1):
        if logits[idx] == best_val:
            token_id = idx
    return 0 if token_id >= REAL_VOCAB else token_id


def test_padded_vocab_device_argmax_numeric_semantics() -> None:
    torch = pytest.importorskip("torch")

    logits = torch.full((3, PADDED_VOCAB), -1000.0)

    # Real-token tie: host torch.argmax chooses the smallest real token id.
    logits[0, 7] = 5.0
    logits[0, 42] = 5.0

    # Padded rows are expected to duplicate token 0 exactly, so the full-VOCAB
    # device scan must still match host argmax over REAL_VOCAB.
    logits[1, 0] = 6.0
    logits[1, REAL_VOCAB:] = logits[1, 0]

    # Defensive clamp case: if a padded row ever wins outright, the device path
    # maps it back to token 0 instead of returning an invalid token id.
    logits[2, 5] = 1.0
    logits[2, REAL_VOCAB + 3] = 9.0

    device_ids = torch.tensor(
        [_device_greedy_argmax_with_clamp(row.tolist()) for row in logits],
        dtype=torch.int64,
    )

    assert torch.equal(device_ids[:2], torch.argmax(logits[:2, :REAL_VOCAB], dim=-1))
    assert device_ids.tolist() == [7, 0, 0]
