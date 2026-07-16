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

from pypto_serving.serving.server import serving_worker


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
        def __init__(self, config, input_queue, output_queue):
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
