# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import logging
import multiprocessing as mp
import os
from pathlib import Path

import torch

from typing import TYPE_CHECKING

from pypto_serving.config.types import (
    DecodeBatch,
    PrefillBatch,
    SamplingParams,
)
from pypto_serving.serving.utils.gc_utils import freeze_gc_heap
from pypto_serving.serving.server.ipc import (
    DecodeRequest,
    NewRequestData,
    PrefillRequest,
    ShutdownCommand,
    StepCommand,
    StepResult,
    decode_command,
    encode_result,
)
from pypto_serving.tools.profile import get_profiler, profile_span

if TYPE_CHECKING:
    from pypto_serving.serving.engine.async_engine import EngineConfig

logger = logging.getLogger(__name__)


class WorkerProcess:
    """Dedicated process that owns a single NPU device and executes model inference.

    Architecture (single-card, extensible to multi-card by spawning multiple workers):
      Main Process  --[input_queue]--> WorkerProcess --[output_queue]--> Main Process
    """

    def __init__(
        self,
        config: EngineConfig,
        input_queue: mp.Queue,
        output_queue: mp.Queue,
    ):
        self.config = config
        self.input_queue = input_queue
        self.output_queue = output_queue

        self.executor = None
        self.sampler = None
        self.model_record = None
        self._page_size: int = 64
        # Request cache: prompt tokens + sampling params registered once per request.
        # Populated by StepCommand.new_requests; entries removed when the request finishes.
        self._req_cache: dict[str, NewRequestData] = {}

    def init_device_and_model(self) -> int:
        from pypto_serving.config.types import ModelRecord
        from pypto_serving.model.common.executor.sampler import Sampler
        from pypto_serving.model.model_loader import ModelLoader

        device_ids = self.config.worker_device_ids()
        device_label = ",".join(str(device_id) for device_id in device_ids)
        pypto_build_dir = self._configure_pypto_build_dir(device_ids)
        if mp.current_process().name != "MainProcess":
            get_profiler(process_name=f"serving-worker-{device_label}")
        with profile_span(
            "WorkerProcess.init_device_and_model",
            cat="worker",
            args={
                "model_id": self.config.model_id,
                "device_id": self.config.device_id,
                "device_ids": list(device_ids),
                "dp_rank": self.config.dp_rank,
                "pypto_build_dir": str(pypto_build_dir),
            },
        ):
            logger.info(
                f"Worker initializing: platform={self.config.platform}, "
                f"devices={list(device_ids)}, dp_rank={self.config.dp_rank}, "
                f"pypto_build_dir={pypto_build_dir}"
            )

            self.sampler = Sampler()

            executor_cls = self._resolve_executor_cls()
            self.executor = executor_cls(
                platform=self.config.platform,
                device_ids=device_ids,
                **self.config.executor_kwargs,
            )

            loaded = ModelLoader().load(
                model_id=self.config.model_id,
                model_dir=self.config.model_dir,
                runtime_config=self.config.runtime_config,
            )

            self.model_record = ModelRecord(
                config=loaded.config,
                runtime=loaded.runtime_model.runtime,
                tokenizer=loaded.tokenizer,
                layer_specs=loaded.layer_specs,
                runtime_model=loaded.runtime_model,
            )

            self._page_size = loaded.runtime_model.runtime.page_size

            register_model = getattr(self.executor, "register_model", None)
            if callable(register_model):
                num_pages = register_model(self.config.model_id, self.model_record)
            else:
                raise RuntimeError("Executor has no register_model method")

            logger.info("Worker model loaded and ready")
            return num_pages

    def _resolve_executor_cls(self):
        if self.config.executor_cls == "PyptoQwen14BExecutor":
            from pypto_serving.model.qwen.npu_executor import Qwen314BPyptoExecutor

            return Qwen314BPyptoExecutor
        if self.config.executor_cls == "PyptoDeepSeekV4Executor":
            from pypto_serving.model.deepseek.npu_executor import DeepSeekV4PyptoExecutor

            return DeepSeekV4PyptoExecutor
        from pypto_serving.model.common.executor.executor import ModelExecutor

        return ModelExecutor

    def _configure_pypto_build_dir(self, device_ids: tuple[int, ...]) -> Path:
        """Give each worker process an isolated PyPTO build base."""
        base = Path(os.environ.get("PYPTO_PROG_BUILD_DIR") or "build_output")
        device_label = "_".join(str(device_id) for device_id in device_ids)
        worker_dir = base / f"serving_dp{self.config.dp_rank}_d{device_label}"
        os.environ["PYPTO_PROG_BUILD_DIR"] = str(worker_dir)
        return worker_dir

    def busy_loop(self) -> None:
        logger.info("Worker entering busy loop")
        while True:
            try:
                raw: bytes = self.input_queue.get()
            except Exception:
                break

            try:
                cmd = decode_command(raw)
            except Exception as e:
                logger.error(f"Worker failed to decode command: {e}", exc_info=True)
                self.output_queue.put(encode_result(StepResult(new_tokens={}, error=str(e))))
                continue

            if isinstance(cmd, ShutdownCommand):
                logger.info("Worker received shutdown command")
                break

            self._handle_step_command(cmd)

        logger.info("Worker exiting")

    def _handle_step_command(self, cmd: StepCommand) -> None:
        """Handle a StepCommand and push an encoded StepResult.

        The whole body is guarded: an exception during request registration or
        device-resource release (steps 1-2) would otherwise propagate out of the
        busy loop and crash the worker. Any failure is reported back to the
        engine as an error result so the loop keeps serving.
        """
        try:
            # 1. Register new requests into the cache.
            for nr in cmd.new_requests:
                self._req_cache[nr.request_id] = nr

            # 2. Release finished requests from device and cache.
            if cmd.finished_request_ids:
                release_finished = getattr(self.executor, "release_finished_requests", None)
                if callable(release_finished):
                    release_finished(cmd.finished_request_ids)
                for req_id in cmd.finished_request_ids:
                    self._req_cache.pop(req_id, None)

            # 3. Execute the step and return the encoded result.
            result = self._execute_step(cmd)
            self.output_queue.put(encode_result(result))
        except Exception as e:
            logger.error(f"Worker step failed: {e}", exc_info=True)
            self.output_queue.put(encode_result(StepResult(new_tokens={}, error=str(e))))

    def _execute_step(self, cmd: StepCommand) -> StepResult:
        """Execute one step using the lightweight IPC protocol."""
        runtime_model = self.model_record.runtime_model
        new_tokens: dict[str, list[int]] = {}

        with profile_span(
            "WorkerProcess.execute_step",
            cat="worker",
            args={"prefill": len(cmd.prefill_requests), "decode": len(cmd.decode_requests)},
        ):
            with self.executor.session():
                if cmd.prefill_requests:
                    self._batch_prefill(cmd.prefill_requests, runtime_model, new_tokens)
                if cmd.decode_requests:
                    self._batch_decode(cmd.decode_requests, runtime_model, new_tokens)

        return StepResult(new_tokens=new_tokens)

    def _batch_prefill(
        self,
        scheduled: list[PrefillRequest],
        runtime_model,
        new_tokens: dict[str, list[int]],
    ) -> None:
        with profile_span(
            "WorkerProcess.batch_prefill",
            cat="worker",
            args={"batch_size": len(scheduled), "request_ids": [pr.request_id for pr in scheduled]},
        ):
            device = runtime_model.runtime.device
            batch_size = len(scheduled)
            max_chunk = max(len(pr.chunk_tokens) for pr in scheduled)

            allow_device_greedy_sampling = (
                self.executor.supports_device_sampling
                and all(self._req_cache[pr.request_id].temperature <= 0.0 for pr in scheduled)
            )

            token_tensor = torch.zeros((batch_size, max_chunk), dtype=torch.long, device=device)
            embeddings = None
            if not self.executor.supports_device_embedding:
                embeddings = torch.zeros(
                    (batch_size, max_chunk, self.model_record.config.hidden_size),
                    dtype=runtime_model.embed_tokens.dtype,
                    device=device,
                )
            positions_tensor = torch.full((batch_size, max_chunk), -1, dtype=torch.long, device=device)

            seq_lens = []
            block_ids_list = []
            for i, pr in enumerate(scheduled):
                row = torch.tensor(pr.chunk_tokens, dtype=torch.long, device=device)
                token_tensor[i, : len(pr.chunk_tokens)] = row
                if embeddings is not None:
                    embeddings[i, : len(pr.chunk_tokens), :] = self.executor.lookup_embeddings(
                        runtime_model, row
                    )
                positions = range(pr.num_computed_tokens, pr.num_computed_tokens + len(pr.chunk_tokens))
                positions_tensor[i, : len(pr.chunk_tokens)] = torch.tensor(
                    list(positions), dtype=torch.long, device=device
                )
                seq_lens.append(pr.num_computed_tokens + len(pr.chunk_tokens))
                block_ids_list.append(pr.block_ids)

            prefill_result = self.executor.run_prefill(
                runtime_model,
                PrefillBatch(
                    request_ids=[pr.request_id for pr in scheduled],
                    token_ids=token_tensor,
                    input_embeddings=embeddings,
                    seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
                    allow_device_greedy_sampling=allow_device_greedy_sampling,
                    positions=positions_tensor,
                    block_ids=block_ids_list,
                ),
            )

            # Sample only for requests whose prefill chunk completes the prompt.
            for i, pr in enumerate(scheduled):
                cached = self._req_cache[pr.request_id]
                # num_prompt_tokens is len(prompt_token_ids), which we have in cache.
                will_be_computed = pr.num_computed_tokens + len(pr.chunk_tokens)
                if will_be_computed >= len(cached.prompt_token_ids):
                    logits = (
                        prefill_result.logits[i]
                        if prefill_result.logits.dim() > 1
                        else prefill_result.logits
                    )
                    params = SamplingParams(
                        temperature=cached.temperature,
                        top_p=cached.top_p,
                        top_k=cached.top_k,
                    )
                    token_id = self._sample_result_row(
                        prefill_result, logits, params, i, allow_device_greedy_sampling
                    )
                    new_tokens[pr.request_id] = [token_id]

    def _batch_decode(
        self,
        scheduled: list[DecodeRequest],
        runtime_model,
        new_tokens: dict[str, list[int]],
    ) -> None:
        with profile_span(
            "WorkerProcess.batch_decode",
            cat="worker",
            args={"batch_size": len(scheduled), "request_ids": [dr.request_id for dr in scheduled]},
        ):
            device = runtime_model.runtime.device

            allow_device_greedy_sampling = (
                self.executor.supports_device_sampling
                and all(self._req_cache[dr.request_id].temperature <= 0.0 for dr in scheduled)
            )

            decode_tokens = [dr.last_token for dr in scheduled]
            prev_tokens = [dr.prev_token for dr in scheduled]
            block_ids_list = [dr.block_ids for dr in scheduled]
            seq_lens = [dr.seq_len for dr in scheduled]

            decode_token_tensor = torch.tensor(decode_tokens, dtype=torch.long, device=device)
            if self.executor.supports_device_embedding:
                # Device kernel embeds directly from token ids — do not build
                # host-side embedding tensors.
                decode_embeddings = None
                prev_embeddings = None
            else:
                decode_embeddings = self.executor.lookup_embeddings(runtime_model, decode_token_tensor)
                prev_token_tensor = torch.tensor(prev_tokens, dtype=torch.long, device=device)
                prev_embeddings = self.executor.lookup_embeddings(runtime_model, prev_token_tensor)

            if self.executor.supports_device_embedding:
                prev_token_tensor = torch.tensor(prev_tokens, dtype=torch.long, device=device)

            decode_result = self.executor.run_decode(
                runtime_model,
                DecodeBatch(
                    request_ids=[dr.request_id for dr in scheduled],
                    token_ids=decode_token_tensor.unsqueeze(1),
                    hidden_states=decode_embeddings,
                    seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
                    allow_device_greedy_sampling=allow_device_greedy_sampling,
                    block_ids=block_ids_list,
                    prev_token_ids=prev_token_tensor,
                    prev_hidden_states=prev_embeddings,
                ),
            )

            for i, dr in enumerate(scheduled):
                cached = self._req_cache[dr.request_id]
                if decode_result.accepted_token_ids is not None:
                    new_tokens[dr.request_id] = list(decode_result.accepted_token_ids[i])
                    continue
                logits = None
                if decode_result.logits is not None:
                    logits = (
                        decode_result.logits[i]
                        if decode_result.logits.dim() > 1
                        else decode_result.logits
                    )
                params = SamplingParams(
                    temperature=cached.temperature,
                    top_p=cached.top_p,
                    top_k=cached.top_k,
                )
                token_id = self._sample_result_row(
                    decode_result, logits, params, i, allow_device_greedy_sampling
                )
                new_tokens[dr.request_id] = [token_id]

    def close(self) -> None:
        """Release executor-owned runtime and device resources."""
        executor = self.executor
        self.executor = None
        if executor is None:
            return

        close = getattr(executor, "close", None)
        if callable(close):
            close()

    def _sample_result_row(
        self,
        result,
        logits: torch.Tensor | None,
        params: SamplingParams,
        row_idx: int,
        allow_device_sampled: bool,
    ) -> int:
        """Return a sampled token from executor output, falling back to host sampling."""
        sampled = getattr(result, "sampled_token_ids", None)
        if allow_device_sampled and sampled is not None:
            flat = sampled.view(-1)
            if flat.numel() <= row_idx:
                raise ValueError(
                    f"sampled_token_ids has {flat.numel()} rows, expected row {row_idx}"
                )
            return int(flat[row_idx].item())
        return self.sampler.sample(logits, params)

def _worker_entry(
    config: EngineConfig,
    input_queue: mp.Queue,
    output_queue: mp.Queue,
    ready_event,
    num_pages_value,
):
    """Entry point for the worker subprocess."""
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for _n in ("simpler_setup", "pypto", "simpler"):
        logging.getLogger(_n).setLevel(logging.WARNING)

    worker = WorkerProcess(config, input_queue, output_queue)
    try:
        num_pages = worker.init_device_and_model()
        num_pages_value.value = num_pages
        # Model weights, compiled kernels and KV-cache objects are now resident.
        # Freeze them so the GC won't rescan them during decode (avoids
        # multi-ms gen2 pauses landing mid-step). Must happen in this process:
        # gc.freeze() does not cross the spawn boundary.
        freeze_gc_heap()
        ready_event.set()
        worker.busy_loop()
    except Exception as e:
        logger.error(f"Worker process failed: {e}", exc_info=True)
        ready_event.set()
    finally:
        try:
            worker.close()
        except Exception:
            logger.exception("Worker process cleanup failed")


def spawn_worker(config: EngineConfig):
    """Spawn a worker process and return (process, input_queue, output_queue, ready_event, num_pages_value).

    ``num_pages_value`` is a shared ``multiprocessing.Value('i')`` that the
    worker writes after ``init_device_and_model()`` completes.  The main
    process reads it to synchronise the ``KvCacheManager`` block metadata with
    the actual device-side KV cache size.
    """
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    ready_event = ctx.Event()
    num_pages_value = ctx.Value("i", 0)

    process = ctx.Process(
        target=_worker_entry,
        args=(config, input_queue, output_queue, ready_event, num_pages_value),
        daemon=False,
    )
    process.start()
    return process, input_queue, output_queue, ready_event, num_pages_value
