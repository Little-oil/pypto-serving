# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Garbage-collection helpers for the serving loop.

A serving process holds a large, permanently-live resident set (model weights,
compiled kernels, KV-cache block objects, tokenizer tables). CPython's cyclic
collector rescans every tracked container on each full (gen2) collection, so
those static objects add a multi-millisecond pause whenever gen2 fires — a
latency spike that can land in the middle of a decode step.

``freeze_gc_heap`` moves the post-warmup heap into the collector's permanent
generation (never rescanned), so subsequent collections only walk per-step
garbage. Call it once per process, after warmup, before serving traffic.
``gc.freeze()`` does not cross a fork/spawn boundary, so each process (engine
and worker) must call it independently.
"""

from __future__ import annotations

import gc
import logging
import os
import time

logger = logging.getLogger(__name__)


def freeze_gc_heap() -> None:
    """Freeze the currently-live heap so the GC stops rescanning static objects.

    Promotes all survivors to the oldest generation, then freezes them into the
    permanent generation. Objects allocated afterward (per-request/per-step)
    are unaffected and remain collectable.
    """
    gc.collect(0)
    gc.collect(1)
    gc.collect(2)
    gc.freeze()
    logger.info("froze %d GC-tracked objects into the permanent generation", gc.get_freeze_count())


# NOTE: not wired into the serving path (kept off the hot loop). To profile GC
# for a future investigation, call maybe_attach_gc_debug_callback() once per
# process right after freeze_gc_heap() — in the worker (serving_worker.py
# `_worker_entry`, before busy_loop()) and/or the engine (async_engine.py
# `ReplicaEngineCore.start`, before the engine loop) — then run with
# PYPTO_SERVING_GC_DEBUG=1 set in that process's environment.
def maybe_attach_gc_debug_callback() -> None:
    """Attach a gc.callbacks hook logging per-collection pause time.

    Enabled by ``PYPTO_SERVING_GC_DEBUG=1``. Logs, for each collection, the
    generation, elapsed milliseconds, and number of objects collected — used to
    confirm whether (and how often) gen2 collections fire under real traffic.
    """
    if os.environ.get("PYPTO_SERVING_GC_DEBUG", "0") != "1":
        return

    state = {"start_ns": 0}

    def _callback(phase: str, info: dict) -> None:
        generation = info.get("generation")
        if generation is None:
            return
        if phase == "start":
            state["start_ns"] = time.monotonic_ns()
        elif phase == "stop":
            elapsed_ms = (time.monotonic_ns() - state["start_ns"]) / 1e6
            logger.info(
                "GC gen%d took %.3fms, collected %s objects",
                generation,
                elapsed_ms,
                info.get("collected", "?"),
            )

    gc.callbacks.append(_callback)
    logger.info("attached GC debug callback (PYPTO_SERVING_GC_DEBUG=1)")
