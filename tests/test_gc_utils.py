# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import gc

from pypto_serving.serving.utils.gc_utils import freeze_gc_heap, maybe_attach_gc_debug_callback


def test_freeze_gc_heap_moves_live_objects_to_permanent_generation():
    """freeze_gc_heap() must leave the frozen count reflecting the live heap so
    those objects are excluded from subsequent collections."""
    # A tracked container that is live across the freeze.
    resident = [[i] for i in range(1000)]  # noqa: F841 - kept alive intentionally
    gc.unfreeze()  # start from a known state
    assert gc.get_freeze_count() == 0

    try:
        freeze_gc_heap()
        frozen = gc.get_freeze_count()
        # The whole live heap (incl. `resident`) is now frozen.
        assert frozen > 1000
        # A collection after freeze does not un-freeze anything.
        gc.collect()
        assert gc.get_freeze_count() == frozen
    finally:
        gc.unfreeze()

    assert gc.get_freeze_count() == 0


def test_gc_debug_callback_attaches_only_when_enabled(monkeypatch):
    before = len(gc.callbacks)

    monkeypatch.delenv("PYPTO_SERVING_GC_DEBUG", raising=False)
    maybe_attach_gc_debug_callback()
    assert len(gc.callbacks) == before  # disabled by default

    monkeypatch.setenv("PYPTO_SERVING_GC_DEBUG", "1")
    maybe_attach_gc_debug_callback()
    try:
        assert len(gc.callbacks) == before + 1
        # The callback must tolerate being invoked (no-op on non-dict phases).
        gc.callbacks[-1]("start", {"generation": 0})
        gc.callbacks[-1]("stop", {"generation": 0, "collected": 3})
        gc.callbacks[-1]("start", {})  # missing generation -> ignored
    finally:
        gc.callbacks.pop()
