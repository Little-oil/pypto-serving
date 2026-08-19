# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run the fixed DeepSeek V4 GBS32 alignment workload under serving profiling."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from pypto_serving import GenerateConfig
from pypto_serving.cli.main import build_parser, build_serving_engine_config
from pypto_serving.model.tokenizer import load_tokenizer
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine
from pypto_serving.tools.profile import configure_profiler, merge_profile, start_profile, stop_profile


REQUEST_COUNT = 32
OUTPUT_TOKENS = 256
ALIGNED_PROMPT = (
    "<｜begin▁of▁sentence｜><｜User｜>"
    "请用中文详细介绍北京故宫，分为历史沿革、整体布局、主要宫殿、建筑特色、重要馆藏、"
    "文化价值和参观建议七节，每节至少一百字，内容准确连贯，不要省略。请使用清晰的小标题，"
    "并说明关键年代、人物与用途。"
    "<｜Assistant｜></think>"
)
EXPECTED_PROMPT_IDS = [
    0, 128803, 2788, 642, 21134, 87336, 6127, 74437, 303, 9969, 5163,
    8689, 4155, 410, 10319, 17996, 410, 2897, 64474, 410, 6786, 10716,
    410, 3036, 6071, 5376, 410, 3415, 87482, 23177, 7383, 3958, 2045,
    303, 1833, 2045, 11732, 21080, 2024, 303, 3975, 12963, 95512, 303,
    4916, 62186, 320, 2788, 2541, 17165, 5968, 24153, 303, 1380, 6977,
    7511, 10776, 410, 13320, 947, 27917, 320, 128804, 128822,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--devices", required=True)
    parser.add_argument("--served-model-name", default="dsv4-flash-w8a8")
    parser.add_argument("--use-compile-cache", action="store_true")
    return parser.parse_args()


async def wait_for_engine_idle(engine: AsyncLLMEngine, timeout: float = 60.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        idle = all(
            not core._batch_queue
            and not core._pending_free_ids
            and not core.scheduler.has_work()
            for core in engine._cores
        )
        if idle and not engine._request_to_replica and engine.pending_token_load() == 0:
            return
        if loop.time() >= deadline:
            raise TimeoutError("engine did not become idle after the warmup batch")
        await asyncio.sleep(0.01)


def serialize_results(results) -> list[dict]:
    return [
        {
            "index": index,
            "text": result.text,
            "token_ids": list(result.token_ids),
            "finish_reason": result.finish_reason,
        }
        for index, result in enumerate(results)
    ]


def validate_results(results: list[dict], label: str) -> None:
    lengths = [len(item["token_ids"]) for item in results]
    if lengths != [OUTPUT_TOKENS] * REQUEST_COUNT:
        raise RuntimeError(f"{label} output lengths are not aligned: {lengths}")
    errors = [item["index"] for item in results if item["finish_reason"] == "error"]
    if errors:
        raise RuntimeError(f"{label} requests failed: {errors}")


async def run(args: argparse.Namespace) -> None:
    artifact_dir = args.artifact_dir.resolve()
    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)

    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if len(devices) != 8 or len(set(devices)) != 8 or any(not item.isdigit() for item in devices):
        raise ValueError(f"--devices must contain exactly eight unique device IDs: {args.devices!r}")

    cli_args = [
        "--model", str(model_dir),
        "--served-model-name", args.served_model_name,
        "--backend", "npu",
        "--platform", "a2a3",
        "--devices", ",".join(devices),
        "--dp", "8",
        "--ep", "8",
        "--tp", "1",
        "--block-size", "128",
        "--max-model-len", "512",
        "--max-num-seqs", "32",
        "--max-num-batched-tokens", "2048",
        "--long-prefill-token-threshold", "128",
        "--speculative-config", '{"method":"mtp","num_speculative_tokens":1}',
        "--temperature", "0",
        "--top-p", "1",
        "--top-k", "0",
        "--no-enable-prefix-caching",
        "--enable-chunked-prefill",
        "--profile",
        "--profile-output", str(artifact_dir / "serving-trace"),
        "--profile-level", "verbose",
    ]
    if args.use_compile_cache:
        cli_args.append("--use-compile-cache")

    parsed = build_parser().parse_args(cli_args)
    engine_config = build_serving_engine_config(parsed)
    if not engine_config.profile_config.enabled:
        raise RuntimeError("verbose serving profiler was not enabled")
    configure_profiler(
        engine_config.profile_config,
        process_name="pypto-gbs32-profile-controller",
        initially_active=False,
    )

    tokenizer = load_tokenizer(model_dir)
    prompt_ids = tokenizer.encode(ALIGNED_PROMPT)
    if prompt_ids != EXPECTED_PROMPT_IDS:
        raise RuntimeError(f"aligned prompt token mismatch: {prompt_ids}")
    generate_config = GenerateConfig(
        max_new_tokens=OUTPUT_TOKENS,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        stream=False,
        ignore_eos=True,
    )
    prompts = [ALIGNED_PROMPT] * REQUEST_COUNT
    engine = AsyncLLMEngine(config=engine_config, tokenizer=tokenizer)

    started = False
    try:
        await engine.start()
        started = True

        warmup = serialize_results(await engine.generate_batch(prompts, generate_config))
        validate_results(warmup, "warmup")
        await wait_for_engine_idle(engine)
        print("Unprofiled aligned GBS32 warmup completed", flush=True)

        controller_started = start_profile()
        if not controller_started:
            raise RuntimeError("failed to start the offline controller profiler")
        workers_started = False
        try:
            await engine.start_profile()
            workers_started = True
            batch_started = time.perf_counter()
            results = await engine.generate_batch(prompts, generate_config)
            elapsed = time.perf_counter() - batch_started
        finally:
            stop_error = None
            if workers_started:
                try:
                    await engine.stop_profile()
                except BaseException as exc:  # preserve the original failure if one exists
                    stop_error = exc
            stop_profile()
            merged_events = merge_profile()
            print(f"PROFILE_MERGED_EVENTS={merged_events}", flush=True)
            if stop_error is not None:
                raise stop_error

        serialized = serialize_results(results)
        validate_results(serialized, "profiled")
        (artifact_dir / "responses.json").write_text(
            json.dumps(serialized, ensure_ascii=False), encoding="utf-8"
        )
        lengths = [len(item["token_ids"]) for item in serialized]
        total_tokens = sum(lengths)
        summary = {
            "batch_elapsed_seconds": elapsed,
            "request_count": REQUEST_COUNT,
            "prompt_tokens_per_request": len(prompt_ids),
            "tokens_per_request": lengths,
            "total_output_tokens": total_tokens,
            "throughput_tokens_per_second": total_tokens / elapsed,
            "effective_tpot_ms": elapsed * 1000.0 / OUTPUT_TOKENS,
            "profiler_enabled": True,
            "profile_level": "verbose",
        }
        (artifact_dir / "performance_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    finally:
        if started:
            await engine.stop()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
