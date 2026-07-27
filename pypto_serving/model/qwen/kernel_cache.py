# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""On-disk cache of compiled Qwen3-14B PyPTO kernels.

Reloading a compiled kernel (IR + device binaries) from disk skips both the JIT
and the ~30s device-binary compile on every launch. Correctness hinges on the
cache key: a kernel binary is specialised by *both*

  * the compile parameters (batch, max_seq, page_size, vocab, head_dim,
    block-table stride, target platform, ...), and
  * the code that fed the compiler (the pypto-lib kernels, the host dispatch
    wrappers in this package, and the pypto compiler itself).

so BOTH must match before a cached slot may be reused. The key is split across
two orthogonal fingerprints:

  * ``params`` -> encoded into the slot marker; a config change is a guaranteed
    MISS, never a stale HIT.
  * ``code``   -> encoded into the slot marker; a kernel/dispatch edit or a
    pypto upgrade is a STALE and forces a recompile + overwrite.

There is one slot per kernel name, overwritten in place, so only the latest
copy is ever kept. All reuse/store is best-effort and never fatal: any failure
falls back to a fresh compile.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

#: Marker file in each slot: ``<code_fp>|<params_fp>`` the slot was built with.
FINGERPRINT_FILE = "fingerprint.txt"
#: Emitted by the pypto compiler; its presence tells us a slot holds a program.
META_FILE = "distributed_meta.json"
#: Sentinel for a fingerprint that could not be computed. It must never compare
#: equal to another sentinel, or genuinely different kernels would collide.
UNKNOWN = "unknown"


def canonical_source(src: str) -> bytes:
    """Return a formatting-insensitive canonical form of Python source.

    Hashing this instead of the raw bytes means cosmetic edits -- blank lines,
    comments, reindentation, spacing -- do NOT invalidate the cache, while any
    change to the parsed program does. Uses the AST (``include_attributes`` off,
    so line/column numbers are excluded); a naive text minify cannot do this
    safely because indentation and keyword/identifier separators are load
    bearing in Python. Falls back to raw bytes for source that does not parse (a
    syntactically broken kernel would not compile anyway).
    """
    try:
        return ast.dump(ast.parse(src), include_attributes=False).encode()
    except SyntaxError:
        return src.encode()


def _pypto_version() -> str:
    try:
        import pypto  # noqa: PLC0415

        return str(getattr(pypto, "__version__", UNKNOWN))
    except Exception:  # noqa: BLE001 - version probe must never be fatal
        return UNKNOWN


@functools.lru_cache(maxsize=None)
def compute_code_fingerprint(pypto_root: str | None) -> str:
    """Fingerprint of everything that feeds codegen for the Qwen3-14B kernels.

    Combines the pypto compiler version with an AST-canonical hash of every
    source that is actually compiled: the pypto-lib Qwen3-14B kernel modules
    *and* the host dispatch wrappers in this package (``qwen3_l3_dispatch.py``),
    which are the functions handed to ``jit_fn.compile``. Hashing the sources
    -- not a submodule SHA -- also catches uncommitted local edits, exactly the
    dev-iteration case the cache must not serve stale.

    Memoised per process: the sources cannot change under a running launch.
    Any probe failure degrades to a marker containing ``UNKNOWN``, which the
    cache treats as a permanent MISS, forcing a safe recompile.
    """
    version = _pypto_version()
    try:
        # Imported lazily to avoid a module import cycle (npu_executor imports
        # this module at top level) and to keep this module torch-free.
        from pypto_serving.model.qwen.npu_executor import _find_pypto_lib_qwen14b_dir  # noqa: PLC0415

        kernel_dir = _find_pypto_lib_qwen14b_dir(pypto_root)
        dispatch = Path(__file__).resolve().parent / "qwen3_l3_dispatch.py"

        sources: list[tuple[str, Path]] = [
            (path.relative_to(kernel_dir).as_posix(), path)
            for path in kernel_dir.rglob("*.py")
        ]
        if dispatch.is_file():
            sources.append((dispatch.name, dispatch))

        digest = hashlib.sha256()
        for rel, path in sorted(sources):
            digest.update(rel.encode())
            digest.update(canonical_source(path.read_text(encoding="utf-8", errors="surrogateescape")))
        return f"{version}+{digest.hexdigest()[:16]}"
    except Exception:  # noqa: BLE001 - fingerprint must never be fatal
        return f"{version}+{UNKNOWN}"


def compute_params_fingerprint(
    name: str,
    dummy_args: Sequence[Any],
    *,
    platform: str,
) -> str:
    """Fingerprint of everything that specialises this kernel's binary.

    The ``dummy_args`` shape+dtype signature encodes every compile-time
    dimension (batch, max_seq, page_size, vocab, head_dim, block-table stride,
    ...); combined with the kernel name and target platform it uniquely
    identifies the compiled kernel. Two launches differing in any of these
    produce different fingerprints, so a config change can never be served a
    stale binary.
    """
    digest = hashlib.sha256()
    digest.update(name.encode())
    digest.update(platform.encode())
    for arg in dummy_args:
        digest.update(str(tuple(arg.shape)).encode())
        digest.update(str(arg.dtype).encode())
    return digest.hexdigest()[:16]


class KernelCache:
    """Load/store compiled kernels under a cache directory (one slot per name)."""

    def __init__(self, cache_dir: str, code_fingerprint: str) -> None:
        self._cache_dir = Path(cache_dir)
        self._code_fingerprint = code_fingerprint

    def _marker(self, params_fingerprint: str) -> str:
        return f"{self._code_fingerprint}|{params_fingerprint}"

    def _usable(self, params_fingerprint: str) -> bool:
        # A failed probe leaves UNKNOWN in the marker. Two UNKNOWN markers must
        # not compare equal (that could HIT genuinely different code), so an
        # unusable fingerprint forces a MISS and a no-op store.
        return UNKNOWN not in self._marker(params_fingerprint)

    def load(
        self,
        name: str,
        params_fingerprint: str,
        *,
        platform: str,
        distributed_config: Any,
    ) -> object | None:
        """Reload a compiled kernel from its slot, or ``None`` on miss/stale/error.

        On a HIT the returned program's ``output_dir`` is the cache slot itself,
        so the L3 worker reuses the cached device binaries and skips recompiling.
        """
        if not self._usable(params_fingerprint):
            logger.info("[kernel-cache] MISS: %s fingerprint unavailable; compiling", name)
            return None
        try:
            from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

            slot = self._cache_dir / name
            if not (slot / META_FILE).exists():
                logger.info("[kernel-cache] MISS: %s not cached under %s; compiling", name, slot)
                return None
            marker_path = slot / FINGERPRINT_FILE
            cached = marker_path.read_text().strip() if marker_path.exists() else None
            current = self._marker(params_fingerprint)
            if cached != current:
                logger.info(
                    "[kernel-cache] STALE: %s cached %r != current %r; recompiling",
                    name, cached, current,
                )
                return None
            compiled = DistributedCompiledProgram.from_dir(
                str(slot), platform=platform, distributed_config=distributed_config,
            )
            logger.info("[kernel-cache] HIT: reused %s from %s", name, slot)
            return compiled
        except Exception as exc:  # noqa: BLE001 - reuse must never be fatal
            logger.warning(
                "[kernel-cache] reuse of %s failed (%s: %s); recompiling",
                name, type(exc).__name__, exc,
            )
            return None

    def store(self, name: str, compiled: Any, params_fingerprint: str) -> None:
        """Persist a freshly-compiled kernel's build dir into its slot.

        Overwrites any existing slot, so only the latest copy is kept. No-op
        when the build dir already IS the slot (reused from cache) or when the
        fingerprint is unavailable. Best-effort: a failure is logged, never
        raised.
        """
        if not self._usable(params_fingerprint):
            return
        try:
            slot = self._cache_dir / name
            src = Path(str(compiled.output_dir))
            if src.resolve() == slot.resolve():
                return  # reused from cache -> already stored
            if not src.exists():
                logger.warning("[kernel-cache] build dir missing for %s; not cached", name)
                return
            slot.parent.mkdir(parents=True, exist_ok=True)
            # Stage into a process-unique temp dir, then swap it in with a single
            # rename: a crash mid-copy (or a concurrent writer) leaves only the
            # .tmp dir, never a half-written slot that would later pass the meta
            # check but fail from_dir. A re-store also self-heals a corrupt slot.
            tmp = slot.with_name(f"{name}.tmp.{os.getpid()}")
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(src, tmp)
            (tmp / FINGERPRINT_FILE).write_text(self._marker(params_fingerprint))
            if slot.exists():
                shutil.rmtree(slot)
            os.replace(tmp, slot)
            logger.info("[kernel-cache] STORED: %s (+ device binaries) -> %s", name, slot)
        except Exception as exc:  # noqa: BLE001 - caching must never be fatal
            logger.warning("[kernel-cache] failed to store %s (%s: %s)", name, type(exc).__name__, exc)
