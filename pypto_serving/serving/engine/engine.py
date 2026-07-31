# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations

import itertools
from collections.abc import Iterator

import torch

from pypto_serving.config.types import (
    DecodeBatch,
    GenerateConfig,
    GenerateResult,
    ModelRecord,
    RequestState,
    RuntimeConfig,
)
from pypto_serving.model.common.executor.executor import ModelExecutor
from pypto_serving.model.common.executor.sampler import Sampler
from pypto_serving.model.model_loader import ModelLoader
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.utils.prefill import pack_prefill_batch
from pypto_serving.tools.profile import profile_span


class LLMEngine:
    """High-level model registry and text generation coordinator."""

    def __init__(
        self,
        model_loader: ModelLoader | None = None,
        kv_cache_manager: KvCacheManager | None = None,
        executor: ModelExecutor | None = None,
        sampler: Sampler | None = None,
    ) -> None:
        """Create an engine from pluggable loader, cache, executor, and sampler."""
        self._model_loader = model_loader or ModelLoader()
        self._kv_cache_manager = kv_cache_manager or KvCacheManager()
        if executor is None:
            raise ValueError("LLMEngine requires a ModelExecutor instance.")
        self._executor = executor
        self._sampler = sampler or Sampler()
        self._models: dict[str, ModelRecord] = {}
        self._request_counter = itertools.count()

    def init_model(
        self,
        model_id: str,
        model_dir: str,
        runtime_config: RuntimeConfig | None = None,
        model_format: str | None = None,
        **loader_options: object,
    ) -> None:
        """Load a model, register its KV cache, and notify the executor."""
        with profile_span("LLMEngine.init_model", cat="engine", args={"model_id": model_id}):
            loaded = self._model_loader.load(
                model_id=model_id,
                model_dir=model_dir,
                runtime_config=runtime_config,
                model_format=model_format,
                **loader_options,
            )
            config = loaded.config
            runtime = loaded.runtime_model.runtime
            record = ModelRecord(
                config=config,
                runtime=runtime,
                tokenizer=loaded.tokenizer,
                layer_specs=loaded.layer_specs,
                runtime_model=loaded.runtime_model,
            )
            register_model = getattr(self._executor, "register_model", None)
            actual_num_pages = None
            if callable(register_model):
                actual_num_pages = register_model(model_id, record)
            self._kv_cache_manager.register_model(
                model_id,
                config,
                runtime,
                num_pages=actual_num_pages,
            )
            self._models[model_id] = record

    def generate(self, model_id: str, prompt: str, config: GenerateConfig | None = None) -> str | Iterator[str]:
        """Generate text for one prompt, optionally returning a text stream."""
        generate_config = config or GenerateConfig()
        if generate_config.stream:
            return self._generate_stream(model_id, prompt, generate_config)
        return self._generate_result(model_id, prompt, generate_config).text

    def _generate_non_stream(self, model_id: str, prompt: str, config: GenerateConfig) -> str:
        """Generate non-streaming text for one prompt."""
        return self._generate_result(model_id, prompt, config).text

    def generate_batch(
        self,
        model_id: str,
        prompts: list[str] | tuple[str, ...],
        config: GenerateConfig | None = None,
    ) -> list[GenerateResult]:
        """Generate non-streaming completions for a batch of prompts."""
        generate_config = config or GenerateConfig()
        if generate_config.stream:
            raise ValueError("generate_batch requires stream=False")
        with profile_span(
            "LLMEngine.generate_batch",
            cat="engine",
            args={
                "model_id": model_id,
                "batch_size": len(prompts),
                "max_new_tokens": generate_config.max_new_tokens,
            },
        ):
            return self._generate_batch_impl(model_id, prompts, generate_config)

    def _generate_batch_impl(
        self,
        model_id: str,
        prompts: list[str] | tuple[str, ...],
        generate_config: GenerateConfig,
    ) -> list[GenerateResult]:
        if not prompts:
            return []
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} is not initialized.")
        record = self._models[model_id]
        if len(prompts) > record.runtime.max_batch_size:
            max_batch_size = record.runtime.max_batch_size
            raise ValueError(
                f"batch has {len(prompts)} prompts, but runtime max_batch_size is {max_batch_size}"
            )

        runtime_model = record.runtime_model
        tokenizer = record.tokenizer
        prompt_token_ids = [tokenizer.encode(prompt) for prompt in prompts]
        for token_ids in prompt_token_ids:
            if not token_ids and record.config.bos_token_id is not None:
                token_ids.append(record.config.bos_token_id)
            if not token_ids:
                raise ValueError("Prompt tokenization produced no tokens.")

        self._executor.validate_generate_batch(record, len(prompts), generate_config)

        requests: list[RequestState] = []
        allocations = []
        try:
            for prompt, token_ids in zip(prompts, prompt_token_ids, strict=True):
                request_id = f"req-{next(self._request_counter)}"
                alloc_len = self._executor.prompt_allocation_length(
                    record,
                    len(token_ids),
                    generate_config,
                )
                alloc = self._kv_cache_manager.allocate_for_prompt(model_id, request_id, alloc_len)
                allocations.append(alloc)
                requests.append(
                    RequestState(
                        request_id=request_id,
                        model_id=model_id,
                        prompt=prompt,
                        prompt_token_ids=token_ids,
                        max_new_tokens=generate_config.max_new_tokens,
                        stop_strings=generate_config.stop,
                        eos_token_id=record.config.eos_token_id,
                        seq_len=len(token_ids),
                        num_prompt_tokens=len(token_ids),
                        kv_allocation=alloc,
                    )
                )

            allow_device_greedy_sampling = (
                generate_config.temperature <= 0.0
                and self._executor.supports_device_sampling
                and self._executor.supports_device_embedding
            )
            embedding_lookup = None
            if not self._executor.supports_device_embedding:
                embedding_lookup = lambda token_ids: self._executor.lookup_embeddings(
                    runtime_model, token_ids
                )

            # Greedy chunked prefill: when total prompt tokens exceed
            # max_num_batched_tokens, split into chunks that each pack up
            # to that many new tokens, filling requests in order.  Only
            # requests with remaining tokens are included in each chunk.
            total_budget = record.runtime.max_num_batched_tokens
            total_tokens = sum(len(ids) for ids in prompt_token_ids)
            if total_tokens > total_budget:
                remaining = [[ri, len(ids)] for ri, ids in enumerate(prompt_token_ids)]
                offsets = [0] * len(prompts)
                completed_logits: dict[int, torch.Tensor] = {}
                completed_sampled_ids: dict[int, torch.Tensor] = {}
                with self._executor.session():
                    while remaining:
                        # greedily take up to total_budget tokens
                        chunk_req: list[int] = []
                        chunk_lens: list[int] = []
                        budget = total_budget
                        for ri, left in remaining:
                            if budget <= 0:
                                break
                            take = min(left, budget)
                            if take <= 0:
                                continue
                            chunk_req.append(ri)
                            chunk_lens.append(take)
                            budget -= take
                            offsets[ri] += take
                        token_chunks: list[list[int]] = []
                        chunk_starts: list[int] = []
                        for row, ri in enumerate(chunk_req):
                            n = chunk_lens[row]
                            start = offsets[ri] - n
                            token_chunks.append(prompt_token_ids[ri][start:offsets[ri]])
                            chunk_starts.append(start)
                        sub_batch = pack_prefill_batch(
                            request_ids=[requests[ri].request_id for ri in chunk_req],
                            token_chunks=token_chunks,
                            seq_lens=[offsets[ri] for ri in chunk_req],
                            chunk_starts=chunk_starts,
                            device=runtime_model.runtime.device,
                            embedding_lookup=embedding_lookup,
                            allow_device_greedy_sampling=allow_device_greedy_sampling,
                            kv_allocations=[allocations[ri] for ri in chunk_req],
                        )
                        prefill_result = self._executor.run_prefill(runtime_model, sub_batch)
                        sampled_ids = (
                            prefill_result.sampled_token_ids.view(-1)
                            if prefill_result.sampled_token_ids is not None
                            else None
                        )
                        for row, ri in enumerate(chunk_req):
                            if offsets[ri] < len(prompt_token_ids[ri]):
                                continue
                            completed_logits[ri] = self._select_batch_row(
                                prefill_result.logits, row
                            ).clone()
                            if sampled_ids is not None:
                                completed_sampled_ids[ri] = sampled_ids[row].clone()
                        # remove completed requests from the pool
                        for i in range(len(remaining) - 1, -1, -1):
                            ri = remaining[i][0]
                            if offsets[ri] >= len(prompt_token_ids[ri]):
                                del remaining[i]
                            else:
                                remaining[i] = [ri, len(prompt_token_ids[ri]) - offsets[ri]]
                if len(completed_logits) != len(requests):
                    raise RuntimeError("Chunked prefill did not produce logits for every request")
                prefill_logits = torch.stack(
                    [completed_logits[request_idx] for request_idx in range(len(requests))]
                )
                prefill_sampled_token_ids = (
                    torch.stack(
                        [completed_sampled_ids[request_idx] for request_idx in range(len(requests))]
                    )
                    if allow_device_greedy_sampling and len(completed_sampled_ids) == len(requests)
                    else None
                )
            else:
                prefill_batch = pack_prefill_batch(
                    request_ids=[request.request_id for request in requests],
                    token_chunks=prompt_token_ids,
                    seq_lens=[len(token_ids) for token_ids in prompt_token_ids],
                    chunk_starts=[0] * len(prompt_token_ids),
                    device=runtime_model.runtime.device,
                    embedding_lookup=embedding_lookup,
                    allow_device_greedy_sampling=allow_device_greedy_sampling,
                    kv_allocations=allocations,
                )
                fast_path_result = self._executor.try_generate_batch(
                    record,
                    requests,
                    prefill_batch,
                    generate_config,
                )
                if fast_path_result is not None:
                    return fast_path_result
                with self._executor.session():
                    prefill_result = self._executor.run_prefill(
                        runtime_model,
                        prefill_batch,
                    )
                prefill_logits = prefill_result.logits
                prefill_sampled_token_ids = (
                    prefill_result.sampled_token_ids
                    if allow_device_greedy_sampling
                    else None
                )

            sampling_params = self._sampler.from_generate_config(generate_config)
            current_tokens = self._sample_batch_rows(
                prefill_logits,
                sampling_params,
                len(requests),
                prefill_sampled_token_ids,
            )
            active_indices = list(range(len(requests)))
            finish_reasons = ["length"] * len(requests)

            for _ in range(generate_config.max_new_tokens):
                next_active: list[int] = []
                decode_tokens: list[int] = []
                for request_idx in active_indices:
                    request = requests[request_idx]
                    current_token = current_tokens[request_idx]
                    request.generated_token_ids.append(current_token)
                    request.output_text = tokenizer.decode(request.generated_token_ids)

                    if record.config.eos_token_id is not None and current_token == record.config.eos_token_id:
                        finish_reasons[request_idx] = "eos"
                        continue
                    if any(stop and request.output_text.endswith(stop) for stop in generate_config.stop):
                        finish_reasons[request_idx] = "stop"
                        continue
                    if len(request.generated_token_ids) >= generate_config.max_new_tokens:
                        finish_reasons[request_idx] = "length"
                        continue

                    alloc = request.kv_allocation
                    if alloc is None:
                        raise RuntimeError("Request is missing KV allocation.")
                    self._kv_cache_manager.ensure_one_more_slot(alloc)
                    request.seq_len += 1
                    next_active.append(request_idx)
                    decode_tokens.append(current_token)

                if not next_active:
                    break

                decode_token_tensor = torch.tensor(
                    decode_tokens,
                    dtype=torch.long,
                    device=runtime_model.runtime.device,
                )
                decode_embeddings = self._decode_embeddings_from_cache_or_lookup(
                    runtime_model,
                    decode_token_tensor,
                )
                active_allocations = []
                for idx in next_active:
                    alloc = requests[idx].kv_allocation
                    if alloc is None:
                        raise RuntimeError("Request is missing KV allocation.")
                    active_allocations.append(alloc)
                decode_result = self._executor.run_decode(
                    runtime_model,
                    DecodeBatch(
                        request_ids=[requests[idx].request_id for idx in next_active],
                        token_ids=decode_token_tensor.unsqueeze(1),
                        hidden_states=decode_embeddings,
                        seq_lens=torch.tensor(
                            [requests[idx].seq_len for idx in next_active],
                            dtype=torch.int32,
                            device=runtime_model.runtime.device,
                        ),
                        allow_device_greedy_sampling=allow_device_greedy_sampling,
                        kv_allocations=active_allocations,
                    ),
                )
                decoded_tokens = self._sample_batch_rows(
                    decode_result.logits,
                    sampling_params,
                    len(next_active),
                    decode_result.sampled_token_ids if allow_device_greedy_sampling else None,
                )
                for row_idx, request_idx in enumerate(next_active):
                    current_tokens[request_idx] = decoded_tokens[row_idx]
                active_indices = next_active
        finally:
            for alloc in allocations:
                self._kv_cache_manager.free(alloc)

        return [
            GenerateResult(
                text=request.output_text,
                token_ids=list(request.generated_token_ids),
                finish_reason=finish_reasons[request_idx],
            )
            for request_idx, request in enumerate(requests)
        ]

    def _generate_stream(self, model_id: str, prompt: str, config: GenerateConfig) -> Iterator[str]:
        """Yield decoded text deltas for one streaming prompt."""
        if model_id not in self._models:
            raise KeyError(f"Model {model_id} is not initialized.")
        record = self._models[model_id]
        runtime_model = record.runtime_model
        tokenizer = record.tokenizer
        prompt_token_ids = tokenizer.encode(prompt)
        if not prompt_token_ids and record.config.bos_token_id is not None:
            prompt_token_ids = [record.config.bos_token_id]
        if not prompt_token_ids:
            raise ValueError("Prompt tokenization produced no tokens.")

        request_id = f"req-{next(self._request_counter)}"
        alloc = self._kv_cache_manager.allocate_for_prompt(model_id, request_id, len(prompt_token_ids))
        request = RequestState(
            request_id=request_id,
            model_id=model_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            max_new_tokens=config.max_new_tokens,
            stop_strings=config.stop,
            eos_token_id=record.config.eos_token_id,
            seq_len=len(prompt_token_ids),
            num_prompt_tokens=len(prompt_token_ids),
            kv_allocation=alloc,
        )

        try:
            embedding_lookup = None
            if not self._executor.supports_device_embedding:
                embedding_lookup = lambda token_ids: self._executor.lookup_embeddings(
                    runtime_model, token_ids
                )

            prefill_batch = pack_prefill_batch(
                request_ids=[request.request_id],
                token_chunks=[prompt_token_ids],
                seq_lens=[len(prompt_token_ids)],
                chunk_starts=[0],
                device=runtime_model.runtime.device,
                embedding_lookup=embedding_lookup,
                kv_allocations=[alloc],
            )

            with self._executor.session():
                prefill_result = self._executor.run_prefill(
                    runtime_model,
                    prefill_batch,
                )

                logits = self._select_batch_row(prefill_result.logits, 0)
                generated: list[int] = []
                emitted_text = ""
                sampling_params = self._sampler.from_generate_config(config)
                current_token = self._sampler.sample(logits, sampling_params)

                for _ in range(config.max_new_tokens):
                    generated.append(current_token)
                    text = tokenizer.decode(generated)
                    delta = text[len(emitted_text) :]
                    emitted_text = text
                    if delta:
                        yield delta
                    if self._should_stop(record, config, generated, emitted_text, current_token):
                        break

                    self._kv_cache_manager.ensure_one_more_slot(alloc)
                    request.seq_len += 1
                    decode_token = torch.tensor([current_token], dtype=torch.long, device=runtime_model.runtime.device)
                    decode_embeddings = self._decode_embeddings_from_cache_or_lookup(
                        runtime_model,
                        decode_token,
                    )
                    decode_result = self._executor.run_decode(
                        runtime_model,
                        DecodeBatch(
                            request_ids=[request.request_id],
                            token_ids=decode_token.unsqueeze(0),
                            hidden_states=decode_embeddings,
                            seq_lens=torch.tensor(
                                [request.seq_len],
                                dtype=torch.int32,
                                device=runtime_model.runtime.device,
                            ),
                            kv_allocations=[alloc],
                        ),
                    )
                    logits = self._select_batch_row(decode_result.logits, 0)
                    current_token = self._sampler.sample(logits, sampling_params)
        finally:
            self._kv_cache_manager.free(alloc)

    def generate_result(self, model_id: str, prompt: str, config: GenerateConfig | None = None) -> GenerateResult:
        """Generate a structured non-streaming result for one prompt."""
        generate_config = config or GenerateConfig()
        if generate_config.stream:
            raise ValueError("generate_result requires stream=False")
        return self._generate_result(model_id, prompt, generate_config)

    def _generate_result(self, model_id: str, prompt: str, config: GenerateConfig) -> GenerateResult:
        """Generate one result by reusing the batch path."""
        return self.generate_batch(model_id, [prompt], config)[0]

    def _sample_batch_rows(
        self,
        logits: torch.Tensor | None,
        sampling_params,
        row_count: int,
        sampled_token_ids: torch.Tensor | None = None,
    ) -> list[int]:
        """Return sampled token IDs, preferring executor-provided device samples."""
        if sampled_token_ids is not None:
            flat_ids = sampled_token_ids.view(-1)
            if flat_ids.numel() < row_count:
                raise ValueError(
                    f"sampled_token_ids has {flat_ids.numel()} rows, expected at least {row_count}"
                )
            return [int(flat_ids[idx].item()) for idx in range(row_count)]
        return [
            self._sampler.sample(
                self._select_batch_row(logits, row_idx),
                sampling_params,
            )
            for row_idx in range(row_count)
        ]

    def _decode_embeddings_from_cache_or_lookup(
        self,
        runtime_model,
        decode_token_tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build decode hidden states only when the executor consumes them."""
        if self._executor.supports_device_decode_embedding:
            return None
        return self._executor.lookup_embeddings(runtime_model, decode_token_tensor)

    @staticmethod
    def _select_batch_row(tensor: torch.Tensor, row_idx: int) -> torch.Tensor:
        """Return row ``row_idx`` from a batch tensor or the tensor itself."""
        return tensor[row_idx] if tensor.dim() > 1 else tensor

    @staticmethod
    def _should_stop(
        record: ModelRecord,
        config: GenerateConfig,
        generated: list[int],
        emitted_text: str,
        current_token: int,
    ) -> bool:
        """Return whether generation should stop for one request."""
        if record.config.eos_token_id is not None and current_token == record.config.eos_token_id:
            return True
        if len(generated) >= config.max_new_tokens:
            return True
        return any(stop and emitted_text.endswith(stop) for stop in config.stop)
