# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Ordered, placement-aware argument container for one L3 dispatch class.

``TaskArgs`` holds every positional argument an L3 callable takes -- host-shared
buffers, static weights, worker-resident handles, and scalars -- and owns the
lifecycle (allocate / stage / clear / build). It replaces the previous
``BufferSet`` (which modelled only I/O buffers) and the per-runner
``values``-dict + ``_mark_resident_args`` + ``ordered_layer_args`` pipeline.

Design rules:

* **Order is the registration order** of ``add_slot`` / ``add_arg`` calls -- there
  is no separate kernel-order tuple. The per-model ``task_args.py`` registers the
  args in the kernel's positional order, so that file is the single source of
  truth for the contract.
* **Each arg declares its kind at registration** (no resident-policy dict):

  - ``add_slot(Slot(...))`` -- a host-shared or device-resident buffer that this
    container allocates / stages / clears.
  - ``add_arg(name, StaticDeviceTensor(t))`` -- a static weight uploaded once and
    cached by the resolver.
  - ``add_arg(name, handle)`` -- an already-resident ``StackedDeviceTensor`` /
    ``DeviceTensor`` (kv cache, device weights), passed through.
  - ``add_arg(name, shared_tensor)`` -- a shared-memory CPU tensor (e.g. a view
    over a slot), passed through.
  - ``add_arg(name, scalar)`` -- a python scalar, passed through.
  - ``add_arg(name, callable)`` -- a zero-arg lazy source for handles that are
    materialized after the worker fork; evaluated at ``build()`` time.

* ``build()`` returns the *unresolved* tuple (markers included). The L3 dispatch
  mixin resolves each arg (``resolve_l3_arg``) and frees per-dispatch uploads.
* The built tuple may be cached (``use_cache=True``). Caching is valid because
  every source is a stable identity: slots are staged in place, handles are
  allocated once, static markers are constant. Call ``reset()`` whenever a
  referenced tensor is reallocated (KV-cache resize, embedding weight).
"""

from __future__ import annotations

from typing import Any

import torch

from .buffer_set import (
    ClearPolicy,
    Placement,
    Slot,
    alloc_device_buffer,
    copy_shared,
    shared_empty,
)

__all__ = ["TaskArgs"]


def _free_device_buffer(worker: Any, buffer: Any) -> None:
    """Release one buffer from ``alloc_device_buffer`` (rollback, best effort)."""
    shards = getattr(buffer, "shards", None)
    if shards is not None:
        worker_ids = getattr(buffer, "worker_ids", None) or range(len(shards))
        for shard, worker_id in zip(shards, worker_ids, strict=False):
            worker.free_tensor(shard, worker_id=worker_id)
    else:
        worker.free_tensor(buffer)


class TaskArgs:
    """Ordered, placement-aware argument container for one L3 dispatch class."""

    def __init__(self, *, stacked: bool = True) -> None:
        # ``stacked`` records the rank model (multi-rank DeepSeek vs single-rank
        # Qwen) for the resolver; it does not affect slot allocation, which is
        # governed by each ``Slot.stacked``.
        self.stacked = stacked
        # Ordered (name, kind, payload) records. kind is "slot" (payload=Slot) or
        # "arg" (payload=value, possibly a lazy callable).
        self._records: list[tuple[str, str, Any]] = []
        self._by_name: dict[str, int] = {}
        self._slots: list[Slot] = []
        self.tensors: dict[str, Any] = {}
        self._host_allocated = False
        self._device_allocated = False
        self._built: tuple[Any, ...] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Arg names in registration order (for diagnostics and fused-name filters)."""
        return tuple(name for name, _, _ in self._records)

    def add_slot(self, slot: Slot) -> TaskArgs:
        """Register a host-shared or device-resident buffer at the next position."""
        self._register_name(slot.name)
        self._records.append((slot.name, "slot", slot))
        self._slots.append(slot)
        self._built = None
        return self

    def add_arg(self, name: str, value: Any) -> TaskArgs:
        """Register a constant arg at the next position.

        ``value`` may be a ``StaticDeviceTensor`` marker, a worker-resident
        handle, a shared-memory CPU tensor, a scalar, or a zero-arg callable
        returning any of those (lazy source for post-fork handles).
        """
        self._register_name(name)
        self._records.append((name, "arg", value))
        self._built = None
        return self

    def _register_name(self, name: str) -> None:
        if name in self._by_name:
            raise ValueError(f"TaskArgs has duplicate arg name: {name!r}")
        self._by_name[name] = len(self._records)

    def allocate_host_shared(self, ctx: Any) -> None:
        """Allocate the host-shared slots. Must run before ``allocate_device``.

        Idempotent: a second call is a no-op (the pre-fork allocation is shared
        with every forked worker, so it must happen exactly once and is safe to
        retry from a hot path that cannot assume whether it has run yet).
        """
        if self._host_allocated:
            return
        if self._device_allocated:
            raise RuntimeError(
                "TaskArgs host-shared slots must be allocated before device-resident slots "
                "(the worker fork inherits host shared memory)"
            )
        for slot in self._slots:
            if slot.placement is Placement.HOST_SHARED:
                self.tensors[slot.name] = shared_empty(slot.shape_fn(ctx), slot.dtype, name=slot.name)
        self._host_allocated = True
        self._built = None

    def allocate_device(self, worker: Any, ctx: Any) -> None:
        """Allocate the device-resident slots once the worker exists.

        Idempotent: the device buffers are owned for the worker's lifetime, so a
        dispatch hot path may call this unconditionally without re-allocating.
        Transactional: if any slot allocation fails, the buffers created during
        this invocation are released and dropped from ``tensors`` so a retry
        cannot orphan the earlier allocations.
        """
        if self._device_allocated:
            return
        created: list[tuple[str, Any]] = []
        try:
            for slot in self._slots:
                if slot.placement is Placement.DEVICE_RESIDENT:
                    self.tensors[slot.name] = alloc_device_buffer(
                        worker, slot.shape_fn(ctx), slot.dtype, stacked=slot.stacked
                    )
                    created.append((slot.name, self.tensors[slot.name]))
        except Exception:
            for name, buffer in created:
                _free_device_buffer(worker, buffer)
                self.tensors.pop(name, None)
            raise
        self._device_allocated = True
        self._built = None

    def stage(self, inputs: dict[str, torch.Tensor]) -> None:
        """Copy per-step ``inputs`` into their host-shared slots."""
        slot_by_name = {slot.name: slot for slot in self._slots}
        for name, src in inputs.items():
            slot = slot_by_name.get(name)
            if slot is None:
                raise KeyError(f"TaskArgs has no slot named {name!r}")
            if slot.placement is not Placement.HOST_SHARED:
                raise TypeError(f"{name!r} is not a host-shared slot and cannot be staged")
            copy_shared(self.tensors[name], src, name=name)

    def clear_outputs(self) -> None:
        """Reset host-shared output slots per their clear policy."""
        for slot in self._slots:
            if slot.clear is ClearPolicy.NONE or slot.placement is not Placement.HOST_SHARED:
                continue
            tensor = self.tensors[slot.name]
            if slot.clear is ClearPolicy.ZERO:
                tensor.zero_()
            else:  # ClearPolicy.FILL_NEG_ONE
                tensor.fill_(-1)

    def build(self, *, use_cache: bool = False) -> tuple[Any, ...]:
        """Return the positional arg tuple in registration order (markers unresolved).

        With ``use_cache=True`` the tuple is memoized; callers must ``reset()`` if
        any referenced tensor is reallocated.
        """
        if use_cache and self._built is not None:
            return self._built
        args: list[Any] = []
        for name, kind, payload in self._records:
            if kind == "slot":
                args.append(self.tensors[name])
            elif callable(payload):
                args.append(payload())
            else:
                args.append(payload)
        result = tuple(args)
        if use_cache:
            self._built = result
        return result

    def reset(self) -> None:
        """Drop the cached built tuple (call on KV-cache resize / device realloc)."""
        self._built = None

    def close(self) -> None:
        """Drop all tensor references and the cache (device tensors are freed by the worker)."""
        self.tensors.clear()
        self._built = None
        self._host_allocated = False
        self._device_allocated = False
