# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from types import SimpleNamespace

import torch

from pypto_serving.config.types import (
    GenerateConfig,
    ModelRecord,
)
from pypto_serving.serving.engine.engine import LLMEngine
from pypto_serving.serving.memory.kv_cache import KvCacheManager

from ..device_sampling_fakes import (
    _CandidateSampler,
    _DeviceSamplingExecutor,
    _DeviceTopkExecutor,
    _FailingSampler,
    _FixedSampler,
    _ImmediateEosExecutor,
    _model,
    _Tokenizer,
    _VariableLengthTokenizer,
)


def test_engine_generate_batch_uses_batched_executor_results():
    model = _model(max_batch_size=2, eos_token_id=0)
    manager = KvCacheManager()
    executor = _ImmediateEosExecutor(manager)
    engine = LLMEngine(kv_cache_manager=manager, executor=executor)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_VariableLengthTokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    results = engine.generate_batch(
        model.config.model_id,
        ["a", "abcd"],
        GenerateConfig(max_new_tokens=2, temperature=0.0),
    )

    assert [result.token_ids for result in results] == [[0], [0]]
    assert [result.finish_reason for result in results] == ["eos", "eos"]
    assert len(executor.prefill_batches) == 1
    prefill_batch = executor.prefill_batches[0]
    assert prefill_batch.token_ids.ndim == 1
    assert prefill_batch.token_ids.tolist() == [1, 1, 2, 3, 4]
    assert prefill_batch.chunk_lens == [1, 4]
    assert prefill_batch.chunk_offsets == [0, 1]
    assert prefill_batch.chunk_starts == [0, 0]
    assert prefill_batch.token_ids.numel() == sum(prefill_batch.chunk_lens)
    assert prefill_batch.input_embeddings is not None
    assert prefill_batch.input_embeddings.shape == (5, model.config.hidden_size)
    assert executor.embedding_lookup_shapes == [(5,)]


def test_engine_chunked_prefill_packs_each_chunk_without_full_prompt_staging():
    model = _model(
        max_batch_size=2,
        eos_token_id=0,
        max_num_batched_tokens=3,
    )
    manager = KvCacheManager()
    executor = _ImmediateEosExecutor(manager)
    engine = LLMEngine(kv_cache_manager=manager, executor=executor)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_VariableLengthTokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    results = engine.generate_batch(
        model.config.model_id,
        ["a", "abcd"],
        GenerateConfig(max_new_tokens=1, temperature=0.0),
    )

    assert [result.token_ids for result in results] == [[0], [0]]
    assert len(executor.prefill_batches) == 2
    first_chunk, second_chunk = executor.prefill_batches
    assert first_chunk.token_ids.tolist() == [1, 1, 2]
    assert first_chunk.seq_lens == [1, 2]
    assert first_chunk.chunk_lens == [1, 2]
    assert first_chunk.chunk_offsets == [0, 1]
    assert first_chunk.chunk_starts == [0, 0]
    assert second_chunk.token_ids.tolist() == [3, 4]
    assert second_chunk.seq_lens == [4]
    assert second_chunk.chunk_lens == [2]
    assert second_chunk.chunk_offsets == [0]
    assert second_chunk.chunk_starts == [2]
    assert executor.embedding_lookup_shapes == [(3,), (2,)]


def test_engine_init_model_uses_executor_reported_kv_capacity():
    model = _model(max_batch_size=2, max_seq_len=128, page_size=64)
    loaded = SimpleNamespace(
        config=model.config,
        runtime_model=model,
        tokenizer=_Tokenizer(),
        layer_specs=[],
    )
    loader = SimpleNamespace(load=lambda **_kwargs: loaded)
    manager = KvCacheManager()
    executor = _ImmediateEosExecutor(manager)
    registered_records = []

    def register_model(model_id, record):
        registered_records.append((model_id, record))
        return 3

    executor.register_model = register_model
    engine = LLMEngine(model_loader=loader, kv_cache_manager=manager, executor=executor)

    engine.init_model(model.config.model_id, "/unused")

    assert manager.num_blocks == 3
    assert manager.num_free_blocks == 3
    assert registered_records == [(model.config.model_id, engine._models[model.config.model_id])]


def test_engine_uses_device_sampled_prefill_token_when_available():
    model = _model(max_batch_size=1, eos_token_id=0)
    model.embed_tokens = torch.arange(
        model.config.vocab_size * model.config.hidden_size, dtype=torch.float32
    ).view(
        model.config.vocab_size,
        model.config.hidden_size,
    )
    manager = KvCacheManager()
    executor = _DeviceSamplingExecutor(manager, first_token=3, second_token=0)
    sampler = _FailingSampler()
    engine = LLMEngine(kv_cache_manager=manager, executor=executor, sampler=sampler)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_Tokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    result = engine.generate_batch(
        model.config.model_id,
        ["abc"],
        GenerateConfig(max_new_tokens=1, temperature=0.0),
    )[0]

    assert result.token_ids == [3]
    assert executor.prefill_calls == 1
    assert executor.decode_calls == 0
    assert sampler.sample_calls == 0


def test_engine_omits_decode_hidden_when_executor_embeds_on_device():
    model = _model(max_batch_size=1, eos_token_id=0)
    model.embed_tokens = torch.arange(
        model.config.vocab_size * model.config.hidden_size, dtype=torch.float32
    ).view(
        model.config.vocab_size,
        model.config.hidden_size,
    )
    manager = KvCacheManager()
    executor = _DeviceSamplingExecutor(manager, first_token=3, second_token=0)
    sampler = _FailingSampler()
    engine = LLMEngine(kv_cache_manager=manager, executor=executor, sampler=sampler)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_Tokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    result = engine.generate_batch(
        model.config.model_id,
        ["abc"],
        GenerateConfig(max_new_tokens=2, temperature=0.0),
    )[0]

    assert result.token_ids == [3, 0]
    assert executor.lookup_calls == 0
    assert executor.decode_calls == 1
    assert executor.decode_hidden_seen[0] is None
    assert sampler.sample_calls == 0


def test_engine_ignores_device_sampled_tokens_for_non_greedy_config():
    model = _model(max_batch_size=1)
    model.embed_tokens = torch.arange(
        model.config.vocab_size * model.config.hidden_size, dtype=torch.float32
    ).view(
        model.config.vocab_size,
        model.config.hidden_size,
    )
    manager = KvCacheManager()
    executor = _DeviceSamplingExecutor(manager, first_token=3, second_token=0)
    sampler = _FixedSampler(token_id=7)
    engine = LLMEngine(kv_cache_manager=manager, executor=executor, sampler=sampler)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_Tokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    result = engine.generate_batch(
        model.config.model_id,
        ["abc"],
        GenerateConfig(max_new_tokens=1, temperature=0.8),
    )[0]

    assert result.token_ids == [7]
    assert executor.prefill_calls == 1
    assert executor.decode_calls == 0
    assert sampler.sample_calls == 1


def test_engine_uses_device_topk_candidates_for_topk_config():
    model = _model(max_batch_size=1)
    manager = KvCacheManager()
    executor = _DeviceTopkExecutor(manager, token_id=7)
    sampler = _CandidateSampler(token_id=7)
    engine = LLMEngine(kv_cache_manager=manager, executor=executor, sampler=sampler)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_Tokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    result = engine.generate_batch(
        model.config.model_id,
        ["abc"],
        GenerateConfig(max_new_tokens=2, temperature=0.8, top_k=4, top_p=1.0),
    )[0]

    assert result.token_ids == [7, 7]
    assert executor.prefill_allow_topk is True
    assert executor.decode_allow_topk is True
    assert sampler.sample_calls == 0
    assert sampler.candidate_calls == 2


def test_engine_chunked_prefill_preserves_device_topk_candidates():
    model = _model(
        max_batch_size=1,
        max_num_batched_tokens=2,
    )
    manager = KvCacheManager()
    executor = _DeviceTopkExecutor(manager, token_id=7)
    sampler = _CandidateSampler(token_id=7)
    engine = LLMEngine(kv_cache_manager=manager, executor=executor, sampler=sampler)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_VariableLengthTokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    result = engine.generate_batch(
        model.config.model_id,
        ["abcd"],
        GenerateConfig(max_new_tokens=1, temperature=0.8, top_k=4, top_p=1.0),
    )[0]

    assert result.token_ids == [7]
    assert executor.prefill_allow_topk is True
    assert sampler.sample_calls == 0
    assert sampler.candidate_calls == 1


def test_engine_skips_device_topk_candidates_without_topk_config():
    model = _model(max_batch_size=1)
    manager = KvCacheManager()
    executor = _DeviceTopkExecutor(
        manager,
        token_id=7,
        always_return_candidates=True,
    )
    sampler = _CandidateSampler(token_id=9)
    engine = LLMEngine(kv_cache_manager=manager, executor=executor, sampler=sampler)
    manager.register_model(model.config.model_id, model.config, model.runtime)
    engine._models[model.config.model_id] = ModelRecord(
        config=model.config,
        runtime=model.runtime,
        tokenizer=_Tokenizer(),
        layer_specs=[],
        runtime_model=model,
    )

    result = engine.generate_batch(
        model.config.model_id,
        ["abc"],
        GenerateConfig(max_new_tokens=1, temperature=0.8, top_k=None),
    )[0]

    assert result.token_ids == [9]
    assert executor.prefill_allow_topk is False
    assert sampler.sample_calls == 1
    assert sampler.candidate_calls == 0
