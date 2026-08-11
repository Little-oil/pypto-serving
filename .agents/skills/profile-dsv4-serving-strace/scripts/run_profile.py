# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run one profiled DeepSeek V4 completion and stop the server cleanly."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[4]
STARTUP_TIMEOUT_SECONDS = 1800
REQUEST_TIMEOUT_SECONDS = 1800
POLL_SECONDS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--use-compile-cache", action="store_true")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--served-model-name", default="dsv4-flash-w8a8")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--prompt", default="Huawei is")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_health(process: subprocess.Popen, port: int) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    next_heartbeat = 0.0
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"server exited during startup with code {return_code}")
        try:
            with opener.open(f"http://127.0.0.1:{port}/health", timeout=5) as response:
                payload = json.loads(response.read())
            if response.status == 200 and payload == {"status": "ok"}:
                print("DeepSeek server is healthy", flush=True)
                return
        except (OSError, TimeoutError, ValueError, urllib.error.URLError) as exc:
            last_error = exc
        now = time.monotonic()
        if now >= next_heartbeat:
            print("Waiting for DeepSeek server startup...", flush=True)
            next_heartbeat = now + 30
        time.sleep(POLL_SECONDS)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def request_completion(
    port: int,
    model_name: str,
    prompt: str,
    max_tokens: int,
) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/completions",
        data=json.dumps(
            {
                "model": model_name,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "top_p": 1.0,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return json.loads(response.read())


def control_profile(port: int, action: str) -> None:
    if action not in {"start", "stop"}:
        raise ValueError(f"unsupported profile action: {action}")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/{action}_profile",
        data=b"",
        method="POST",
    )
    with opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        if response.status != 200:
            raise RuntimeError(f"{action}_profile returned HTTP {response.status}")
    suffix = "started" if action == "start" else "stopped"
    print(f"SA profiler {suffix}", flush=True)


def stop_server(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.kill(process.pid, signal.SIGINT)
        process.wait(timeout=90)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=30)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except OSError:
        pass
    try:
        process.wait(timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        pass


def print_log_tail(server_log: Path) -> None:
    if not server_log.exists():
        return
    content = server_log.read_bytes()[-50000:].decode(errors="replace")
    print("\n--- server.log tail ---", flush=True)
    print(content, flush=True)


def main(args: argparse.Namespace) -> int:
    artifact_dir = args.artifact_dir
    server_log = artifact_dir / "server.log"
    if not args.model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {args.model_dir}")
    if args.max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")
    if server_log.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite existing run: {server_log}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    device_list = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(device_list) != 8 or len(set(device_list)) != 8:
        raise ValueError(f"--devices must contain exactly eight unique IDs, got {args.devices!r}")
    if any(not item.isdigit() for item in device_list):
        raise ValueError(f"--devices must contain non-negative integer IDs, got {args.devices!r}")
    devices = ",".join(device_list)
    parallel_size = len(device_list)

    port = unused_local_port()
    command = [
        sys.executable,
        str(SCRIPT_DIR / "launch_server.py"),
        "--model",
        str(args.model_dir),
        "--served-model-name",
        args.served_model_name,
        "--backend",
        "npu",
        "--platform",
        "a2a3",
        "--devices",
        devices,
        "--dp",
        str(parallel_size),
        "--ep",
        str(parallel_size),
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
        "--profile",
        "--profile-output",
        str(artifact_dir / "serving-trace"),
        "--profile-level",
        "verbose",
    ]
    if args.use_compile_cache:
        command.append("--use-compile-cache")
    print(f"Server command: {' '.join(command)}", flush=True)
    print(f"Server log: {server_log}", flush=True)
    with server_log.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        try:
            wait_for_health(process, port)
            control_profile(port, "start")
            request_failure: BaseException | None = None
            try:
                started = time.perf_counter()
                response = request_completion(
                    port,
                    args.served_model_name,
                    args.prompt,
                    args.max_tokens,
                )
                elapsed = time.perf_counter() - started
                print(f"Completion elapsed_s={elapsed:.6f}", flush=True)
                print(f"Completion response: {json.dumps(response, ensure_ascii=False)}", flush=True)
                choices = response.get("choices", [])
                if len(choices) != 1:
                    raise AssertionError(f"expected one choice, got {choices!r}")
                usage = response.get("usage") or {}
                if usage.get("completion_tokens") != args.max_tokens:
                    raise AssertionError(
                        f"expected {args.max_tokens} completion tokens, got usage={usage!r}"
                    )
                (artifact_dir / "completion-response.json").write_text(
                    json.dumps(response, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                (artifact_dir / "completion.txt").write_text(
                    choices[0].get("text", ""),
                    encoding="utf-8",
                )
            except BaseException as exc:
                request_failure = exc
                raise
            finally:
                try:
                    control_profile(port, "stop")
                except Exception as exc:
                    if request_failure is None:
                        raise
                    print(f"WARNING: failed to stop SA profiler: {exc}", flush=True)
        finally:
            print("Stopping server gracefully...", flush=True)
            stop_server(process)
    print("Server stopped", flush=True)
    return 0


if __name__ == "__main__":
    parsed_args = parse_args()
    parsed_args.artifact_dir = parsed_args.artifact_dir.resolve()
    server_log_path = parsed_args.artifact_dir / "server.log"
    try:
        raise SystemExit(main(parsed_args))
    except BaseException:
        print_log_tail(server_log_path)
        raise
