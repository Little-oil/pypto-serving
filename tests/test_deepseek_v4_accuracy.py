# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""DeepSeek V4 HTTP generation accuracy guard for CI."""

from __future__ import annotations

import io
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "dsv4-flash-w8a8"
PROMPT = "Huawei is"
MAX_NEW_TOKENS = 10
EXPECTED_TEXT = " a leading global information and communications technology (ICT)"

STARTUP_TIMEOUT_SECONDS = 600
OVERALL_TIMEOUT_SECONDS = int(os.environ.get("PYPTO_DSV4_OVERALL_TIMEOUT_SECONDS", "1650"))
HEARTBEAT_SECONDS = 30


def _task_devices() -> tuple[int, ...]:
    raw_devices = os.environ.get("TASK_DEVICE", "")
    try:
        devices = tuple(int(value.strip()) for value in raw_devices.split(",") if value.strip())
    except ValueError:
        pytest.fail(f"TASK_DEVICE must contain comma-separated integer device IDs, got {raw_devices!r}")
    if len(devices) != 8 or len(set(devices)) != 8 or any(device < 0 for device in devices):
        pytest.fail(f"TASK_DEVICE must contain exactly 8 unique non-negative device IDs, got {raw_devices!r}")
    return devices


def _unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(model_dir: Path, devices: tuple[int, ...], port: int) -> list[str]:
    # Keep these serving options aligned with docs/dev/model/deepseek-v4.md.
    # CI substitutes only the checkpoint, task-submit devices, and free port.
    return [
        sys.executable,
        "-m",
        "pypto_serving.cli",
        "--model",
        str(model_dir),
        "--served-model-name",
        MODEL_ID,
        "--backend",
        "npu",
        "--platform",
        "a2a3",
        "--devices",
        ",".join(str(device) for device in devices),
        "--dp",
        "1",
        "--tp",
        "8",
        "--block-size",
        "128",
        "--max-model-len",
        "260",
        "--max-num-seqs",
        "1",
        "--max-num-batched-tokens",
        "512",
        "--long-prefill-token-threshold",
        "2048",
        "--enable-mtp",
        "--no-enable-prefix-caching",
        "--port",
        str(port),
        "--show-startup-logs",
    ]


def _wait_for_health(process: subprocess.Popen, port: int, deadline: float) -> None:
    url = f"http://127.0.0.1:{port}/health"
    startup_deadline = min(deadline, time.monotonic() + STARTUP_TIMEOUT_SECONDS)
    next_heartbeat = time.monotonic()
    last_error: BaseException | None = None

    while time.monotonic() < startup_deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"DeepSeek server exited before becoming healthy (code={return_code})")
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                payload = json.loads(response.read())
            if response.status == 200 and payload == {"status": "ok"}:
                print("DeepSeek server is healthy", flush=True)
                return
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as exc:
            last_error = exc

        now = time.monotonic()
        if now >= next_heartbeat:
            print("Waiting for DeepSeek server startup...", flush=True)
            next_heartbeat = now + HEARTBEAT_SECONDS
        time.sleep(2)

    raise TimeoutError(f"DeepSeek server did not become healthy: {last_error}")


def _request_completion(process: subprocess.Popen, port: int, deadline: float) -> dict:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(
            {
                "model": MODEL_ID,
                "prompt": PROMPT,
                "max_tokens": MAX_NEW_TOKENS,
                "temperature": 0.0,
                "top_p": 1.0,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    results: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def send_request() -> None:
        try:
            timeout = max(1.0, deadline - time.monotonic())
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8")
                results.put((True, json.loads(body)))
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_body = "<failed to read error body>"
            results.put(
                (False, RuntimeError(f"completion request returned HTTP {exc.code}: {error_body}"))
            )
        except BaseException as exc:
            results.put((False, exc))

    threading.Thread(target=send_request, name="deepseek-completion", daemon=True).start()
    while time.monotonic() < deadline:
        try:
            succeeded, value = results.get(timeout=HEARTBEAT_SECONDS)
        except queue.Empty:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"DeepSeek server exited during generation (code={return_code})"
                ) from None
            print("Waiting for DeepSeek completion...", flush=True)
            continue
        if succeeded:
            if not isinstance(value, dict):
                raise TypeError(f"completion response must be a JSON object, got {type(value).__name__}")
            return value
        if isinstance(value, BaseException):
            raise value
        raise RuntimeError(f"completion request failed: {value}")
    raise TimeoutError("DeepSeek completion exceeded the end-to-end timeout")


def _stop_process_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except OSError as exc:
        print(f"WARNING: failed to terminate process group {process.pid}: {exc}", flush=True)
        return

    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"WARNING: process group {process.pid} still alive after SIGKILL", flush=True)
        except Exception as exc:
            print(f"WARNING: failed to reap process group {process.pid}: {exc}", flush=True)
        return
    except Exception as exc:
        print(f"WARNING: failed to wait for process group {process.pid}: {exc}", flush=True)
        return

    # The server parent may exit before a worker child. Give the process group a
    # short grace period, then kill any remaining descendants.
    shutdown_deadline = time.monotonic() + 2
    while time.monotonic() < shutdown_deadline:
        try:
            os.killpg(process.pid, 0)
        except OSError:
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass


def _print_server_log(log_path: Path) -> None:
    if not log_path.exists():
        return
    try:
        with log_path.open("rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            log_file.seek(max(0, log_file.tell() - 50000))
            content = log_file.read().decode("utf-8", errors="replace")
    except OSError as exc:
        print(f"WARNING: failed to read DeepSeek server log: {exc}", flush=True)
        return
    print("\n--- DeepSeek server log (tail) ---", flush=True)
    print(content, flush=True)


def test_deepseek_v4_http_completion_matches_expected_text(tmp_path: Path) -> None:
    model_dir_env = os.environ.get("PYPTO_DSV4_MODEL_DIR")
    model_dir = Path(model_dir_env) if model_dir_env else None
    if model_dir is None or not model_dir.is_dir():
        pytest.fail(f"PYPTO_DSV4_MODEL_DIR not set or not a directory: {model_dir}")
    devices = _task_devices()
    port = _unused_local_port()
    log_path = tmp_path / "deepseek-v4-server.log"
    deadline = time.monotonic() + OVERALL_TIMEOUT_SECONDS

    try:
        with log_path.open("w", encoding="utf-8") as server_log:
            process = subprocess.Popen(
                _server_command(model_dir, devices, port),
                cwd=ROOT,
                stdout=server_log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            try:
                _wait_for_health(process, port, deadline)
                response = _request_completion(process, port, deadline)
                print(f"DeepSeek completion response: {response}", flush=True)

                assert response.get("model") == MODEL_ID
                choices = response.get("choices")
                assert isinstance(choices, list) and len(choices) == 1
                assert choices[0].get("text") == EXPECTED_TEXT
                assert choices[0].get("finish_reason") == "length"
            finally:
                _stop_process_group(process)
    except BaseException:
        _print_server_log(log_path)
        raise


def test_completion_http_error_includes_response_body(monkeypatch) -> None:
    error = urllib.error.HTTPError(
        "http://127.0.0.1/completions",
        500,
        "Internal Server Error",
        {},
        io.BytesIO(b"device allocation failed"),
    )

    def raise_http_error(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(urllib.request, "urlopen", raise_http_error)

    class RunningProcess:
        @staticmethod
        def poll():
            return None

    with pytest.raises(RuntimeError, match="HTTP 500: device allocation failed"):
        _request_completion(RunningProcess(), 1, time.monotonic() + 1)


def test_stop_process_group_suppresses_final_wait_timeout(monkeypatch, capsys) -> None:
    class StuckProcess:
        pid = 123

        @staticmethod
        def wait(timeout):
            raise subprocess.TimeoutExpired("server", timeout)

    monkeypatch.setattr(os, "killpg", lambda *_args: None)

    _stop_process_group(StuckProcess())

    assert "still alive after SIGKILL" in capsys.readouterr().out


def test_print_server_log_reads_only_tail(tmp_path, capsys) -> None:
    log_path = tmp_path / "server.log"
    log_path.write_bytes(b"excluded-prefix\n" + b"x" * 60000 + b"\nincluded-tail\n")

    _print_server_log(log_path)

    output = capsys.readouterr().out
    assert "excluded-prefix" not in output
    assert "included-tail" in output
