# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import signal
from types import SimpleNamespace

import pytest

from pypto_serving.serving.engine.async_engine import (
    ReplicaEngineCore,
)
from pypto_serving.serving.memory.kv_cache import KvCacheManager
from pypto_serving.serving.sched.scheduler import (
    Request,
    RequestStatus,
    ScheduledRequest,
    SchedulerOutput,
)
from pypto_serving.serving.server import serving_worker
from pypto_serving.serving.server.ipc import (
    NewRequestData,
    PrefillRequest,
    StepCommand,
    StepResult,
    decode_command,
    encode_command,
)
from pypto_serving.serving.server.serving_worker import WorkerProcess

from ..device_sampling_fakes import _FixedSampler, _ImmediateEosExecutor, _model


def test_step_command_preserves_grouped_cache_metadata_on_preempted_restart():
    core = ReplicaEngineCore.__new__(ReplicaEngineCore)
    core._worker_known_req_ids = {"req"}
    request = Request(
        request_id="req",
        prompt_token_ids=[1, 2],
        max_new_tokens=1,
        status=RequestStatus.RUNNING,
    )
    scheduled = ScheduledRequest(
        request=request,
        num_new_tokens=2,
        is_prefill=True,
        block_ids_by_group={"ori": [3, 4], "state": [5]},
        cache_partition=2,
    )
    output = SchedulerOutput(scheduled_requests=[scheduled])

    command = core._build_step_command(output, finished_ids=["req"])
    decoded = decode_command(encode_command(command))

    assert [item.request_id for item in decoded.new_requests] == ["req"]
    assert decoded.finished_request_ids == ["req"]
    assert decoded.prefill_requests[0].block_ids_by_group == {
        "ori": [3, 4],
        "state": [5],
    }
    assert decoded.prefill_requests[0].cache_partition == 2


def test_partitioned_prefill_chunks_keep_cache_partitions_unique():
    requests = [
        PrefillRequest(
            request_id=request_id,
            chunk_tokens=[1],
            num_computed_tokens=0,
            block_ids=[],
            cache_partition=partition,
        )
        for request_id, partition in (("a", 0), ("b", 0), ("c", 1))
    ]

    chunks = WorkerProcess._partitioned_prefill_chunks(requests, max_batch=2)

    assert [[request.request_id for request in chunk] for chunk in chunks] == [
        ["a", "c"],
        ["b"],
    ]


def test_worker_releases_preempted_state_before_same_command_reregistration():
    released: list[str] = []
    results: list[bytes] = []
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = SimpleNamespace(release_finished_requests=released.extend)
    worker._req_cache = {
        "req": NewRequestData("req", [0], 0.0, 1.0, None),
    }
    worker._last_tokens = {}
    worker.output_queue = SimpleNamespace(put=results.append)
    worker._execute_step = lambda _cmd: StepResult(new_tokens={})
    replacement = NewRequestData("req", [1, 2], 0.0, 1.0, None)
    command = StepCommand(
        new_requests=[replacement],
        prefill_requests=[],
        decode_requests=[],
        finished_request_ids=["req"],
    )

    worker._handle_step_command(command)

    assert released == ["req"]
    assert worker._req_cache["req"] == replacement
    assert len(results) == 1


def test_serving_worker_packs_variable_length_prefill_chunks():
    model = _model(max_batch_size=2, eos_token_id=0)
    manager = KvCacheManager()
    executor = _ImmediateEosExecutor(manager)
    worker = WorkerProcess.__new__(WorkerProcess)
    worker.executor = executor
    worker.sampler = _FixedSampler(token_id=0)
    worker.model_record = SimpleNamespace(config=model.config)
    worker._req_cache = {
        "long": NewRequestData(
            request_id="long",
            prompt_token_ids=[1, 2, 3, 4],
            temperature=0.0,
            top_p=1.0,
            top_k=None,
        ),
        "short": NewRequestData(
            request_id="short",
            prompt_token_ids=[5],
            temperature=0.0,
            top_p=1.0,
            top_k=None,
        ),
    }
    scheduled = [
        PrefillRequest(
            request_id="long",
            chunk_tokens=[2, 3, 4],
            num_computed_tokens=1,
            block_ids=[0],
        ),
        PrefillRequest(
            request_id="short",
            chunk_tokens=[5],
            num_computed_tokens=0,
            block_ids=[1],
        ),
    ]
    new_tokens: dict[str, list[int]] = {}

    worker._batch_prefill(scheduled, model, new_tokens)

    assert new_tokens == {"long": [0], "short": [0]}
    assert len(executor.prefill_batches) == 1
    prefill_batch = executor.prefill_batches[0]
    assert prefill_batch.token_ids.ndim == 1
    assert prefill_batch.token_ids.tolist() == [2, 3, 4, 5]
    assert prefill_batch.seq_lens == [4, 1]
    assert prefill_batch.chunk_lens == [3, 1]
    assert prefill_batch.chunk_offsets == [0, 3]
    assert prefill_batch.chunk_starts == [1, 0]
    assert prefill_batch.token_ids.numel() == sum(prefill_batch.chunk_lens)
    assert prefill_batch.input_embeddings is not None
    assert prefill_batch.input_embeddings.shape == (4, model.config.hidden_size)
    assert executor.embedding_lookup_shapes == [(4,)]


def test_worker_close_releases_executor_once():
    executor = SimpleNamespace(close_calls=0)

    def close():
        executor.close_calls += 1

    executor.close = close
    worker = serving_worker.WorkerProcess.__new__(serving_worker.WorkerProcess)
    worker.executor = executor

    worker.close()
    worker.close()

    assert executor.close_calls == 1
    assert worker.executor is None


@pytest.mark.parametrize("busy_loop_fails", [False, True])
def test_worker_entry_always_closes_worker(monkeypatch, busy_loop_fails):
    calls = SimpleNamespace(close=0, ready=0)

    class FakeWorker:
        def __init__(self, config, input_queue, output_queue, profile_output_queue=None):
            pass

        def init_device_and_model(self):
            return 7

        def busy_loop(self):
            if busy_loop_fails:
                raise RuntimeError("worker failed")

        def close(self):
            calls.close += 1

    monkeypatch.setattr(serving_worker, "WorkerProcess", FakeWorker)
    monkeypatch.setattr(signal, "signal", lambda *_args: None)
    ready_event = SimpleNamespace(set=lambda: setattr(calls, "ready", calls.ready + 1))
    num_pages_value = SimpleNamespace(value=0)

    serving_worker._worker_entry(
        SimpleNamespace(),
        SimpleNamespace(),
        SimpleNamespace(),
        ready_event,
        num_pages_value,
    )

    assert num_pages_value.value == 7
    assert calls.ready >= 1
    assert calls.close == 1
