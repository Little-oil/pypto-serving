# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
import time
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field, replace
from typing import Callable

from pypto_serving.config.parallel import ParallelConfig
from pypto_serving.config.types import RuntimeConfig
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.utils.gc_utils import freeze_gc_heap
from pypto_serving.serving.sched.scheduler import (
    Request,
    RequestStatus,
    Scheduler,
    SchedulerConfig,
    SchedulerOutput,
)
from pypto_serving.serving.server.ipc import (
    DecodeRequest,
    NewRequestData,
    PrefillRequest,
    ShutdownCommand,
    StepCommand,
    decode_result,
    encode_command,
)
from pypto_serving.serving.server.serving_worker import spawn_worker
from pypto_serving.tools.profile import profile_instant, profile_span

logger = logging.getLogger(__name__)
_DEFAULT_WORKER_INIT_TIMEOUT_SECONDS = 1800.0
_DEFAULT_WORKER_STEP_TIMEOUT_SECONDS = 300.0
_DEFAULT_DEEPSEEK_V4_WORKER_STEP_TIMEOUT_SECONDS = 1200.0


def _positive_env_timeout_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive number of seconds") from exc
    if timeout <= 0:
        raise ValueError(f"{name} must be a positive number of seconds")
    return timeout


def _worker_init_timeout_seconds() -> float:
    return _positive_env_timeout_seconds("PYPTO_WORKER_INIT_TIMEOUT", _DEFAULT_WORKER_INIT_TIMEOUT_SECONDS)


def _worker_step_timeout_seconds(executor_cls: str = "") -> float:
    default = _DEFAULT_WORKER_STEP_TIMEOUT_SECONDS
    if executor_cls == "PyptoDeepSeekV4Executor":
        default = _DEFAULT_DEEPSEEK_V4_WORKER_STEP_TIMEOUT_SECONDS
    return _positive_env_timeout_seconds("SERVING_WORKER_STEP_TIMEOUT", default)


@dataclass
class EngineConfig:
    # Model
    model_id: str = ""
    model_dir: str = ""

    # Device / executor
    platform: str = "a2a3"
    device_id: int = 0
    device_ids: tuple[int, ...] = ()
    parallel_config: ParallelConfig | None = None
    dp_rank: int = 0
    executor_cls: str = "PyptoQwen14BExecutor"
    executor_kwargs: dict = field(default_factory=dict)

    # Runtime
    runtime_config: RuntimeConfig | None = None

    # Scheduler / serving
    max_num_running_reqs: int = 32
    max_num_scheduled_tokens: int = 4096
    long_prefill_token_threshold: int = 2048
    engine_loop_interval: float = 0.001

    # Feature flags
    enable_prefix_cache: bool = True
    enable_chunk_prefill: bool = True

    def worker_device_ids(self) -> tuple[int, ...]:
        """Return the device ids this engine worker should own."""
        if self.parallel_config is not None:
            groups = self.parallel_config.replica_device_groups
            if len(groups) == 1:
                return groups[0]
            if 0 <= self.dp_rank < len(groups):
                return groups[self.dp_rank]
            raise ValueError(
                f"dp_rank {self.dp_rank} is outside configured replica groups: "
                f"{len(groups)}"
            )
        if self.device_ids:
            return tuple(int(device) for device in self.device_ids)
        return (int(self.device_id),)


@dataclass
class _RequestContext:
    request: Request
    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # When False (non-streaming), intermediate TokenOutputs are suppressed and
    # only the final one is enqueued — one queue push / one HTTP wake-up per
    # request instead of one per token. Stop-string detection still runs every
    # step; only publishing is deferred (cf. vLLM's FINAL_ONLY output kind).
    stream: bool = True
    # Incremental-detokenization state (avoids re-decoding the full output each
    # step, which would be O(N^2) over a generation). detok_text is the running
    # cumulative text; the offsets bound the per-step decode window.
    detok_text: str = ""
    detok_prefix_offset: int = 0
    detok_read_offset: int = 0


@dataclass
class TokenOutput:
    token_id: int | None = None
    text: str = ""
    finished: bool = False
    finish_reason: str = ""


class ReplicaEngineCore:
    """Engine core for one serving replica.

    A core owns all mutable serving state for one replica: one scheduler, one
    KV cache manager, one worker process, one executor/model runtime, and one
    tensor-parallel device group. Requests assigned to this core are scheduled
    only against this core's local KV cache and worker state.
    """

    def __init__(
        self,
        config: EngineConfig,
        tokenizer=None,
        eos_token_id: int | None = None,
        bos_token_id: int | None = None
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id

        runtime = self.config.runtime_config or RuntimeConfig()
        block_size = runtime.page_size
        self._runtime = runtime
        # Block metadata is initialised lazily after the worker reports the
        # actual device-side KV cache page count (computed from remaining
        # NPU memory after model weight upload).
        self.kv_cache_manager = KvCacheManager(
            num_blocks=None,
            block_size=block_size,
            enable_prefix_cache=self.config.enable_prefix_cache,
        )
        self.kv_cache_manager.init_groups(
            runtime.kv_cache_groups,
            max_batch_size=runtime.max_batch_size,
        )

        scheduler_config = SchedulerConfig(
            max_num_running_reqs=self.config.max_num_running_reqs,
            max_num_scheduled_tokens=self.config.max_num_scheduled_tokens,
            long_prefill_token_threshold=self.config.long_prefill_token_threshold,
            max_seq_len=runtime.max_seq_len,
            enable_prefix_cache=self.config.enable_prefix_cache,
            enable_chunk_prefill=self.config.enable_chunk_prefill,
            num_speculative_tokens=runtime.num_speculative_tokens,
        )
        self.scheduler = Scheduler(config=scheduler_config, kv_cache_manager=self.kv_cache_manager)

        self._request_contexts: dict[str, _RequestContext] = {}
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._request_counter = 0
        self._pending_free_ids: list[str] = []

        self._worker_process = None
        self._input_queue = None
        self._output_queue = None
        # Tracks which request_ids the worker has already received via
        # NewRequestData — prompt tokens are sent exactly once per request.
        self._worker_known_req_ids: set[str] = set()

    async def start(self) -> None:
        """Start worker process and engine loop."""
        with profile_span("AsyncLLMEngine.start", cat="serving"):
            process, input_q, output_q, ready_event, num_pages_value = spawn_worker(self.config)
            self._worker_process = process
            self._input_queue = input_q
            self._output_queue = output_q

            logger.info("Waiting for worker to initialize model...")
            try:
                init_timeout = _worker_init_timeout_seconds()
                ready = await asyncio.to_thread(ready_event.wait, timeout=init_timeout)
                if not ready:
                    raise RuntimeError(
                        f"Worker failed to initialize within {init_timeout:g}s timeout; "
                        "set PYPTO_WORKER_INIT_TIMEOUT to allow more time for large checkpoints"
                    )
            except BaseException:
                await asyncio.to_thread(self._shutdown_worker, timeout=5)
                raise
            logger.info("Worker ready")

            # Synchronise block metadata with the actual device-side KV cache size.
            actual_num_pages = num_pages_value.value
            if actual_num_pages <= 0:
                raise RuntimeError(
                    f"Worker reported invalid KV cache page count: {actual_num_pages}"
                )
            if self.kv_cache_manager.has_groups:
                logger.info(
                    "Grouped KV cache pools initialised: %s",
                    ", ".join(
                        f"{name}={self.kv_cache_manager.group_num_blocks(name)}"
                        for name in self.kv_cache_manager.group_names
                    ),
                )
            else:
                self.kv_cache_manager._init_blocks(actual_num_pages, self._runtime.page_size)
                logger.info(
                    "KV cache block pool initialised: num_blocks=%d, block_size=%d",
                    actual_num_pages,
                    self._runtime.page_size,
                )

        # The KV-cache block pool, scheduler tables and tokenizer are now
        # resident. Freeze the engine-process heap so the GC won't rescan them
        # during serving (see gc_utils). Per-process: the worker freezes its
        # own heap separately.
        freeze_gc_heap()

        self._running = True
        self._loop_task = asyncio.create_task(self._engine_loop())
        logger.info("ReplicaEngineCore started")

    async def stop(self) -> None:
        """Stop engine loop and worker process."""
        self._running = False
        if self._loop_task is not None:
            await self._loop_task
            self._loop_task = None

        await asyncio.to_thread(self._shutdown_worker, timeout=30)
        logger.info("ReplicaEngineCore stopped")

    def generate_request_id(self) -> str:
        self._request_counter += 1
        return f"serving-req-{self._request_counter}"

    def pending_token_load(self) -> int:
        """Estimate unfinished work for routing new data-parallel requests."""
        load = 0
        for request in self.scheduler.requests.values():
            if request.status.is_finished:
                continue
            prompt_remaining = max(0, request.num_prompt_tokens - request.num_computed_tokens)
            generation_remaining = max(0, request.max_new_tokens - len(request.output_token_ids))
            load += prompt_remaining + generation_remaining
        return load

    async def add_request(
        self,
        request_id: str,
        prompt: str,
        config,
        *,
        on_queued: Callable[[], None] | None = None,
        prompt_token_ids: Sequence[int] | None = None,
    ) -> AsyncGenerator[TokenOutput, None]:
        """Add a request and yield token outputs as they are generated."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is required for request processing")
        with profile_span(
            "ReplicaEngineCore.add_request",
            cat="serving",
            args={"request_id": request_id, "max_new_tokens": config.max_new_tokens},
        ):
            if prompt_token_ids is None:
                prompt_token_ids = self.tokenizer.encode(prompt)
            if not prompt_token_ids and self.bos_token_id is not None:
                prompt_token_ids = [self.bos_token_id]
            if not prompt_token_ids:
                raise ValueError("Prompt tokenization produced no tokens.")

            request = Request(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                max_new_tokens=config.max_new_tokens,
                arrival_time=time.time(),
                stop_strings=tuple(config.stop) if config.stop else (),
                eos_token_id=self.eos_token_id,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
            )

            ctx = _RequestContext(request=request, stream=getattr(config, "stream", True))
            self._request_contexts[request_id] = ctx
            self.scheduler.add_request(request)
            logger.info(
                "request %s received: prompt=%d tokens, max_new_tokens=%d",
                request_id, len(prompt_token_ids), config.max_new_tokens,
            )
            if on_queued is not None:
                on_queued()
            profile_instant(
                "request.queued",
                cat="serving",
                args={"request_id": request_id, "prompt_tokens": len(prompt_token_ids)},
            )

        finished_normally = False
        try:
            while True:
                output: TokenOutput = await ctx.queue.get()
                yield output
                if output.finished:
                    finished_normally = True
                    e2e = time.time() - request.arrival_time
                    n_out = len(request.output_token_ids)
                    logger.info(
                        "request %s finished: prompt=%d out=%d reason=%s e2e=%.2fs (%.1f tok/s)",
                        request_id, len(prompt_token_ids), n_out, output.finish_reason,
                        e2e, (n_out / e2e) if e2e > 0 else 0.0,
                    )
                    break
        finally:
            # Only cancellation/disconnect needs cleanup here. On normal
            # completion the request already finished in the scheduler and
            # _process_step_output already scheduled the worker free, so
            # re-scheduling would double-release: the id may have been drained
            # into a StepCommand before this finally runs, defeating a plain
            # membership check and freeing the same request on the worker twice.
            if not finished_normally and request_id in self._request_contexts:
                self._request_contexts.pop(request_id, None)
                self.scheduler.abort_request(request_id)
                # Aborted/cancelled ids must ride the next StepCommand's
                # finished_request_ids, otherwise they leak in _req_cache /
                # _worker_known_req_ids and pin device resources.
                self._schedule_worker_free(request_id)

    async def abort_request(self, request_id: str) -> None:
        ctx = self._request_contexts.pop(request_id, None)
        if ctx is None:
            # Already finished/cleaned up: nothing pinned to release, and the
            # scheduler no longer tracks it. Avoid scheduling a duplicate free.
            return
        self.scheduler.abort_request(request_id)
        await ctx.queue.put(
            TokenOutput(finished=True, finish_reason="FINISHED_ABORTED")
        )
        # See note in add_request's finally block: schedule worker-side cleanup.
        self._schedule_worker_free(request_id)

    def _schedule_worker_free(self, request_id: str) -> None:
        """Queue a request id for worker-side release on the next StepCommand.

        Idempotent against ids still queued; combined with the single-owner
        cleanup paths (normal completion vs. abort/cancel) this guarantees each
        request is released on the worker exactly once.
        """
        if request_id not in self._pending_free_ids:
            self._pending_free_ids.append(request_id)

    async def _engine_loop(self) -> None:
        """Main loop: schedule -> send to worker -> receive results -> dispatch."""
        logger.info("Engine loop started")
        while self._running:
            if not self.scheduler.has_work():
                # No schedulable work, but a just-aborted request may still be
                # pinned on the worker. Flush pending frees now instead of
                # waiting for unrelated future work to carry them.
                await self._flush_pending_frees()
                await asyncio.sleep(self.config.engine_loop_interval)
                continue

            with profile_span("scheduler.schedule", cat="scheduler"):
                scheduler_output = self.scheduler.schedule()
            if scheduler_output.is_empty:
                await self._flush_pending_frees()
                await asyncio.sleep(self.config.engine_loop_interval)
                continue

            finished_ids = self._pending_free_ids.copy()
            self._pending_free_ids.clear()
            with profile_span(
                "scheduler.queue_worker_step",
                cat="scheduler",
                args={"scheduled": len(scheduler_output.scheduled_requests)},
            ):
                step_cmd = self._build_step_command(scheduler_output, finished_ids)
                self._input_queue.put(encode_command(step_cmd))

            try:
                with profile_span("scheduler.wait_worker_output", cat="scheduler"):
                    step_timeout = _worker_step_timeout_seconds(self.config.executor_cls)
                    raw_output = await asyncio.to_thread(
                        self._output_queue.get, timeout=step_timeout
                    )
            except queue.Empty:
                logger.error(f"Worker response timed out ({step_timeout:g}s)")
                self._handle_step_error(scheduler_output)
                continue

            step_result = decode_result(raw_output)
            error = step_result.error
            # Unwrap list[int] values back to int | list[int] for update_from_output.
            new_tokens: dict[str, int | list[int]] = {
                req_id: (tokens[0] if len(tokens) == 1 else tokens)
                for req_id, tokens in step_result.new_tokens.items()
            }

            if error:
                logger.error(f"Worker returned error: {error}")
                self._handle_step_error(scheduler_output)
                continue

            with profile_span(
                "scheduler.process_step_output",
                cat="scheduler",
                args={"new_tokens": len(new_tokens)},
            ):
                self._process_step_output(scheduler_output, new_tokens)

        logger.info("Engine loop stopped")

    def _build_step_command(
        self,
        scheduler_output: SchedulerOutput,
        finished_ids: list[str],
    ) -> StepCommand:
        """Build a lightweight StepCommand from the scheduler output.

        Prompt tokens for requests that the worker has not yet seen are shipped
        as ``NewRequestData`` entries exactly once; subsequent steps carry only
        per-request deltas (~1 KB total at batch 16).
        """
        new_requests: list[NewRequestData] = []
        prefill_requests: list[PrefillRequest] = []
        decode_requests: list[DecodeRequest] = []

        for sr in scheduler_output.scheduled_requests:
            req = sr.request
            req_id = req.request_id

            # Register with worker the first time this request is scheduled.
            if req_id not in self._worker_known_req_ids:
                new_requests.append(NewRequestData(
                    request_id=req_id,
                    prompt_token_ids=list(req.prompt_token_ids),
                    temperature=req.temperature,
                    top_p=req.top_p,
                    top_k=req.top_k,
                ))
                self._worker_known_req_ids.add(req_id)

            if sr.is_prefill:
                num_computed = sr.num_computed_tokens
                num_new = sr.num_new_tokens
                chunk_tokens = req.prompt_token_ids[num_computed: num_computed + num_new]
                prefill_requests.append(PrefillRequest(
                    request_id=req_id,
                    chunk_tokens=list(chunk_tokens),
                    num_computed_tokens=num_computed,
                    block_ids=list(sr.block_ids),
                ))
            else:
                output_ids = req.output_token_ids
                prompt_ids = req.prompt_token_ids
                last_token = output_ids[-1] if output_ids else prompt_ids[-1]
                if len(output_ids) >= 2:
                    prev_token = output_ids[-2]
                elif output_ids and prompt_ids:
                    prev_token = prompt_ids[-1]
                else:
                    prev_token = last_token
                decode_requests.append(DecodeRequest(
                    request_id=req_id,
                    last_token=last_token,
                    prev_token=prev_token,
                    seq_len=req.num_tokens,
                    block_ids=list(sr.block_ids),
                ))

        # Remove finished requests from the known-set so they are re-registered
        # if the same request_id is ever reused (unlikely but correct).
        for req_id in finished_ids:
            self._worker_known_req_ids.discard(req_id)

        return StepCommand(
            new_requests=new_requests,
            prefill_requests=prefill_requests,
            decode_requests=decode_requests,
            finished_request_ids=finished_ids,
        )

    async def _flush_pending_frees(self) -> None:
        """Send a cleanup-only StepCommand when frees are pending but no work is
        schedulable, so an aborted request's worker cache / device slot is not
        pinned until unrelated future work happens to carry the free along.

        The worker tolerates empty prefill/decode batches and replies with an
        empty StepResult, which we drain to keep the request/response queues in
        lock-step with the normal loop.
        """
        if not self._pending_free_ids:
            return

        finished_ids = self._pending_free_ids.copy()
        self._pending_free_ids.clear()
        for req_id in finished_ids:
            self._worker_known_req_ids.discard(req_id)

        cleanup_cmd = StepCommand(
            new_requests=[],
            prefill_requests=[],
            decode_requests=[],
            finished_request_ids=finished_ids,
        )
        self._input_queue.put(encode_command(cleanup_cmd))
        try:
            step_timeout = _worker_step_timeout_seconds(self.config.executor_cls)
            raw_output = await asyncio.to_thread(
                self._output_queue.get, timeout=step_timeout
            )
        except queue.Empty:
            logger.error(f"Worker cleanup-step timed out ({step_timeout:g}s)")
            return
        step_result = decode_result(raw_output)
        if step_result.error:
            logger.error(f"Worker cleanup-step returned error: {step_result.error}")

    def _handle_step_error(self, scheduler_output: SchedulerOutput) -> None:
        """On worker error, abort all requests in the failed batch."""
        for sr in scheduler_output.scheduled_requests:
            request_id = sr.request.request_id
            ctx = self._request_contexts.get(request_id)
            if ctx is not None:
                ctx.queue.put_nowait(
                    TokenOutput(finished=True, finish_reason="error")
                )
            self._schedule_worker_free(request_id)
            self.scheduler.abort_request(request_id)

    def _process_step_output(
        self,
        scheduler_output: SchedulerOutput,
        new_tokens: dict[str, int | list[int]],
    ) -> None:
        """Process worker results: update scheduler state, push tokens to request queues."""
        request_outputs = self.scheduler.update_from_output(scheduler_output, new_tokens)

        for req_output in request_outputs:
            ctx = self._request_contexts.get(req_output.request_id)
            if ctx is None:
                continue

            text = self._detokenize_incrementally(ctx)

            if not req_output.finished and ctx.request.stop_strings:
                for stop in ctx.request.stop_strings:
                    if stop and text.endswith(stop):
                        req_output.finished = True
                        req_output.finish_reason = "FINISHED_STOP"
                        self.scheduler.finish_request(
                            req_output.request_id, RequestStatus.FINISHED_STOP
                        )
                        break

            if req_output.finished:
                # Flush the authoritative full decode: if generation ends while a
                # multi-token character is incomplete (or a token legitimately
                # decodes to U+FFFD), the incremental path withholds that tail
                # forever. A one-shot full decode at finish guarantees the final
                # text matches the offline baseline instead of being truncated.
                text = self._finalize_detokenization(ctx)
                self._schedule_worker_free(req_output.request_id)

            # Non-streaming requests only need the final output: suppress
            # intermediate ones to save a queue push and HTTP-coroutine wake-up
            # per token. Detok + stop detection above still ran this step, so the
            # final text is complete.
            if not ctx.stream and not req_output.finished:
                continue

            token_output = TokenOutput(
                token_id=req_output.new_token_id,
                text=text,
                finished=req_output.finished,
                finish_reason=req_output.finish_reason,
            )
            ctx.queue.put_nowait(token_output)

    def _detokenize_incrementally(self, ctx: _RequestContext) -> str:
        """Decode only the newly-completed text and append it to the running text.

        O(1) amortized per step (bounded decode window) instead of re-decoding
        the full output_token_ids every step (O(N^2) over a generation).
        Returns the cumulative decoded text so far.
        """
        output_ids = ctx.request.output_token_ids
        if not output_ids:
            return ctx.detok_text

        # Decode a short window: [prefix_offset:] gives context so the delta is
        # rendered identically to a full decode; the delta is the tail beyond
        # what [prefix_offset:read_offset] already covered.
        prefix_ids = output_ids[ctx.detok_prefix_offset: ctx.detok_read_offset]
        new_ids = output_ids[ctx.detok_prefix_offset:]

        prefix_text = self.tokenizer.decode(prefix_ids) if prefix_ids else ""
        new_text = self.tokenizer.decode(new_ids)

        if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
            # No new complete text yet (e.g. mid multi-token character); wait for
            # more tokens without advancing offsets.
            return ctx.detok_text

        delta = new_text[len(prefix_text):]
        ctx.detok_text += delta
        # Keep a small sliding context window (last few tokens) rather than
        # collapsing the prefix onto the read offset. A 1-2 token prefix loses
        # boundary context and can corrupt spacing / multi-token characters for
        # SentencePiece / byte-level BPE tokenizers.
        ctx.detok_read_offset = len(output_ids)
        ctx.detok_prefix_offset = max(0, ctx.detok_read_offset - 3)
        return ctx.detok_text

    def _finalize_detokenization(self, ctx: _RequestContext) -> str:
        """Return the authoritative final text for a finished request.

        The incremental path withholds a trailing U+FFFD (an incomplete
        multi-token character) waiting for a token that never arrives once
        generation stops. A single full decode of the whole output at finish
        matches the offline baseline; O(N) once per request is negligible.
        """
        output_ids = ctx.request.output_token_ids
        if not output_ids:
            return ctx.detok_text
        final_text = self.tokenizer.decode(output_ids)
        ctx.detok_text = final_text
        ctx.detok_read_offset = len(output_ids)
        ctx.detok_prefix_offset = max(0, ctx.detok_read_offset - 3)
        return final_text

    def _shutdown_worker(self, *, timeout: float) -> None:
        input_q = self._input_queue
        process = self._worker_process

        if input_q is not None:
            with contextlib.suppress(Exception):
                # New protocol: send encoded ShutdownCommand bytes.
                input_q.put(encode_command(ShutdownCommand()))

        if process is not None:
            with contextlib.suppress(Exception):
                process.join(timeout=timeout)
            with contextlib.suppress(Exception):
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=1)

        self._worker_process = None
        self._input_queue = None
        self._output_queue = None


class AsyncLLMEngine:
    """Async serving engine that routes requests across replica cores.

    The engine owns one or more ``ReplicaEngineCore`` instances and exposes the
    server-facing async API: ``start``, ``stop``, ``add_request``,
    ``abort_request``, and ``generate_request_id``. With one serving replica it
    wraps a single core. With multiple replicas it selects a core for each
    request and records request placement so aborts reach the correct replica.
    """

    def __init__(
        self,
        config: EngineConfig,
        tokenizer=None,
        eos_token_id: int | None = None,
        bos_token_id: int | None = None,
        *,
        core_factory: Callable[..., ReplicaEngineCore] = ReplicaEngineCore,
    ) -> None:
        parallel = config.parallel_config
        if parallel is None:
            worker_devices = config.worker_device_ids()
            parallel = ParallelConfig(
                tensor_parallel_size=len(worker_devices),
                devices=worker_devices,
            )
            config = replace(config, parallel_config=parallel)

        self.config = config
        self.tokenizer = tokenizer
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.parallel_config = parallel
        self._request_counter = 0
        self._route_counter = 0
        self._request_to_replica: dict[str, int] = {}
        self._route_extra_load = [0 for _ in parallel.replica_device_groups]
        self._cores: list[ReplicaEngineCore] = []

        for dp_rank, device_group in enumerate(parallel.replica_device_groups):
            replica_parallel = parallel.for_replica(device_group)
            replica_config = replace(
                config,
                device_id=device_group[0],
                parallel_config=replica_parallel,
                dp_rank=dp_rank,
            )
            self._cores.append(
                core_factory(
                    config=replica_config,
                    tokenizer=tokenizer,
                    eos_token_id=eos_token_id,
                    bos_token_id=bos_token_id,
                )
            )

    async def start(self) -> None:
        """Start all DP engine cores in parallel."""
        tasks = [asyncio.create_task(core.start()) for core in self._cores]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop all DP engine cores."""
        await asyncio.gather(*(core.stop() for core in reversed(self._cores)))

    def generate_request_id(self) -> str:
        self._request_counter += 1
        return f"serving-req-{self._request_counter}"

    def pending_token_load(self) -> int:
        return sum(core.pending_token_load() for core in self._cores)

    @property
    def scheduler(self) -> Scheduler:
        return self._single_core().scheduler

    @property
    def kv_cache_manager(self) -> KvCacheManager:
        return self._single_core().kv_cache_manager

    async def add_request(
        self,
        request_id: str,
        prompt: str,
        config,
    ) -> AsyncGenerator[TokenOutput, None]:
        replica_idx = self._select_replica()
        prompt_token_ids = self._tokenize_prompt(prompt)
        request_load = self._estimate_request_load(prompt_token_ids, config)
        self._route_extra_load[replica_idx] += request_load
        self._request_to_replica[request_id] = replica_idx
        route_extra_active = True

        def clear_route_extra_load() -> None:
            nonlocal route_extra_active
            if not route_extra_active:
                return
            self._route_extra_load[replica_idx] = max(
                0,
                self._route_extra_load[replica_idx] - request_load,
            )
            route_extra_active = False

        try:
            core = self._cores[replica_idx]
            async for output in core.add_request(
                request_id,
                prompt,
                config,
                on_queued=clear_route_extra_load,
                prompt_token_ids=prompt_token_ids,
            ):
                yield output
        finally:
            self._request_to_replica.pop(request_id, None)
            clear_route_extra_load()

    async def abort_request(self, request_id: str) -> None:
        replica_idx = self._request_to_replica.get(request_id)
        if replica_idx is not None:
            await self._cores[replica_idx].abort_request(request_id)
            return
        for core in self._cores:
            await core.abort_request(request_id)

    def _select_replica(self) -> int:
        loads = [
            core.pending_token_load() + self._route_extra_load[idx]
            for idx, core in enumerate(self._cores)
        ]
        replica_count = len(self._cores)
        ordered = [
            (loads[idx], (idx - self._route_counter) % replica_count, idx)
            for idx in range(replica_count)
        ]
        replica_idx = min(ordered)[2]
        self._route_counter = (replica_idx + 1) % replica_count
        return replica_idx

    def _single_core(self):
        if len(self._cores) != 1:
            raise AttributeError("scheduler and kv_cache_manager are only exposed for single-replica engines")
        return self._cores[0]

    def _tokenize_prompt(self, prompt: str) -> Sequence[int] | None:
        if self.tokenizer is not None:
            prompt_token_ids = self.tokenizer.encode(prompt)
            if not prompt_token_ids and self.bos_token_id is not None:
                prompt_token_ids = [self.bos_token_id]
            if not prompt_token_ids:
                raise ValueError("Prompt tokenization produced no tokens.")
            return prompt_token_ids
        return None

    def _estimate_request_load(self, prompt_token_ids: Sequence[int] | None, config) -> int:
        prompt_tokens = len(prompt_token_ids) if prompt_token_ids is not None else 0
        return prompt_tokens + int(getattr(config, "max_new_tokens", 0))
