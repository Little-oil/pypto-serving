# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared primitives for L3 worker input/output buffers.

Both the DeepSeek and Qwen runners stage tensors into a shared L3 worker that is
forked *after* host-side shared-memory allocation: a forked chip worker can only
see host memory inherited at fork, so every host tensor must live in shared memory
before the worker starts. This module collects the upload markers, shared-memory
allocators, and the arg resolver the two runners previously duplicated.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from pypto.runtime import DeviceTensor, StackedDeviceTensor


@dataclass
class StaticDeviceTensor:
    """A CPU tensor marker uploaded to every chip worker once and cached.

    ``cache_state`` flags tensors whose device mirror holds mutable per-step cache
    state, so the runner knows to free them on reset. Single-rank runners leave it
    at the default.
    """

    tensor: torch.Tensor
    cache_state: bool = False


@dataclass
class TransientDeviceTensor:
    """A CPU tensor marker uploaded for one dispatch and then freed."""

    tensor: torch.Tensor


def share_cpu_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return ``tensor`` as a contiguous shared-memory CPU tensor."""
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    if not tensor.is_shared():
        tensor = tensor.share_memory_()
    return tensor


def shared_empty(shape: Sequence[int], dtype: torch.dtype, *, name: str = "") -> torch.Tensor:
    """Allocate an uninitialized shared-memory CPU tensor.

    ``name`` is accepted for caller context and is currently unused (kept so the
    model runners can label allocations without per-call-site changes).
    """
    del name
    return torch.empty(tuple(int(dim) for dim in shape), dtype=dtype).share_memory_()


def copy_shared(dst: torch.Tensor, src: torch.Tensor, *, name: str) -> None:
    """Copy ``src`` into the pre-allocated shared buffer ``dst``.

    Validates shape/dtype match and short-circuits when both views share storage.
    """
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
    if dst.data_ptr() == src.data_ptr():
        return
    dst.copy_(src)


def resolve_l3_arg(
    worker: Any,
    arg: Any,
    static_cache: dict[Any, Any],
    *,
    uploaded: list[Any] | None = None,
    cache_keys: set[Any] | None = None,
    stacked: bool = True,
) -> Any:
    """Resolve one L3 dispatch arg into its worker-resident form.

    - ``StaticDeviceTensor`` is uploaded once and cached by ``(data_ptr, shape, dtype)``;
      ``alloc_stacked_tensor`` when ``stacked`` (multi-rank), else ``alloc_tensor``.
    - ``TransientDeviceTensor`` is uploaded and appended to ``uploaded`` for the caller
      to free after the dispatch.
    - A non-shared CPU tensor -- bare or wrapped in an upload marker -- raises
      (it cannot survive the worker fork).
    - Anything else (scalars, already-shared CPU tensors, raw device handles) passes through.
    """
    # Validate before the upload-marker branches: a marker-wrapped tensor must
    # satisfy the same shared-CPU rule as a bare tensor, or it cannot survive
    # the worker fork.
    upload = arg.tensor if isinstance(arg, (StaticDeviceTensor, TransientDeviceTensor)) else arg
    if isinstance(upload, torch.Tensor) and upload.device.type == "cpu" and not upload.is_shared():
        raise TypeError(
            "L3 dispatch requires shared-memory CPU tensors allocated before the "
            f"worker starts; got non-shared tensor shape={tuple(upload.shape)} dtype={upload.dtype}"
        )
    if isinstance(arg, StaticDeviceTensor):
        tensor = arg.tensor
        key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
        cached = static_cache.get(key)
        if cached is None:
            cached = (
                worker.alloc_stacked_tensor(tensor)
                if stacked
                else worker.alloc_tensor(tensor.shape, tensor.dtype, init=tensor)
            )
            static_cache[key] = cached
        if arg.cache_state and cache_keys is not None:
            cache_keys.add(key)
        return cached
    if isinstance(arg, TransientDeviceTensor):
        tensor = arg.tensor
        dev = worker.alloc_tensor(tensor.shape, tensor.dtype, init=tensor)
        if uploaded is not None:
            uploaded.append(dev)
        return dev
    return arg


def ordered_layer_args(values: dict[str, Any], names: Sequence[str]) -> tuple[Any, ...]:
    """Return ``values`` projected onto the kernel's positional ``names`` order."""
    missing = [name for name in names if name not in values]
    if missing:
        raise KeyError(f"layer dispatch is missing tensors: {', '.join(missing)}")
    return tuple(values[name] for name in names)


class Placement(Enum):
    """Where a buffer slot lives."""

    HOST_SHARED = "host_shared"  #: shared-memory CPU tensor, allocated before the worker fork
    DEVICE_RESIDENT = "device_resident"  #: worker-resident tensor, allocated after the fork


class ClearPolicy(Enum):
    """How a host-shared output slot is reset before each dispatch."""

    NONE = "none"
    ZERO = "zero"
    FILL_NEG_ONE = "fill_neg_one"


@dataclass(frozen=True)
class Slot:
    """One named I/O buffer and how it is allocated.

    ``shape_fn`` receives the model-provided layout/context and returns the full
    tensor shape (for multi-rank ``stacked`` slots the leading dim is the rank
    count). ``clear`` only applies to host-shared output slots.
    """

    name: str
    placement: Placement
    dtype: torch.dtype
    shape_fn: Callable[[Any], tuple[int, ...]]
    clear: ClearPolicy = ClearPolicy.NONE
    stacked: bool = True


def alloc_device_buffer(
    worker: Any,
    full_shape: Sequence[int],
    dtype: torch.dtype,
    *,
    stacked: bool,
) -> Any:
    """Allocate an uninitialized worker-resident buffer.

    For ``stacked`` (multi-rank) buffers, allocates one shard per worker and
    returns a ``StackedDeviceTensor``; otherwise a single ``DeviceTensor``.
    Mirrors DeepSeek's ``_alloc_empty_stacked_tensor``.
    """
    full_shape = tuple(int(dim) for dim in full_shape)
    if not stacked:
        return worker.alloc_tensor(full_shape, dtype)
    worker_ids = tuple(range(full_shape[0]))
    shards: list[DeviceTensor] = []
    try:
        for worker_id in worker_ids:
            shards.append(worker.alloc_tensor(full_shape[1:], dtype, worker_id=worker_id))
    except Exception:
        for shard, worker_id in zip(shards, worker_ids, strict=False):
            worker.free_tensor(shard, worker_id=worker_id)
        raise
    return StackedDeviceTensor(shards, full_shape, worker_ids)
