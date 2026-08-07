# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Run DeepSeek V4 Flash W8A8 generation without starting an HTTP server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

from pypto_serving import GenerateConfig
from pypto_serving.config.parallel import parse_device_ids
from pypto_serving.config.types import GenerateResult
from pypto_serving.model.deepseek.offline import build_deepseek_v4_offline_engine_config
from pypto_serving.model.tokenizer import load_tokenizer
from pypto_serving.serving.engine.async_engine import AsyncLLMEngine
from pypto_serving.tools.profile import (
    configure_profiler,
    create_profile_config,
    merge_profile,
    start_profile,
    stop_profile,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline DeepSeek V4 Flash W8A8 generation on eight Ascend NPUs."
    )
    parser.add_argument("--model-dir", required=True, help="Local W8A8 compressed-tensors checkpoint.")
    parser.add_argument("--prompt", required=True, help="Prompt text.")
    parser.add_argument("--model-id", default=None, help="Result model id; defaults to the directory name.")
    parser.add_argument("--platform", default="a2a3", choices=["a2a3sim", "a2a3", "a5sim", "a5"])
    parser.add_argument(
        "--devices",
        default="0,1,2,3,4,5,6,7",
        help="Exactly eight comma-separated NPU ids used by overlapped attention DP=8 / MoE EP=8.",
    )
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=32, help="Maximum active requests.")
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=512,
        help="Maximum tokens scheduled in one prefill iteration.",
    )
    parser.add_argument("--long-prefill-token-threshold", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=1, help="Replicate --prompt for batched generation.")
    parser.add_argument("--npu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--dtype", default="bfloat16", help="Runtime weight dtype metadata.")
    parser.add_argument("--kv-cache-dtype", default="bfloat16")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--stop", action="append", default=[], help="Stop string; may be repeated.")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument(
        "--enable-mtp",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable DeepSeek V4 MTP speculative decoding.",
    )
    parser.add_argument(
        "--enable-chunked-prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--kernel-cache-dir", default=None)
    parser.add_argument("--save-kernels-dir", default=None)
    parser.add_argument("--pypto-root", default=None)
    parser.add_argument("--profile", action="store_true", help="Capture generation-time SA_PROFILE events.")
    parser.add_argument("--profile-output", default="./profile_out/deepseek-v4-offline")
    parser.add_argument("--profile-level", default="e2e,kernel")
    return parser


def _print_result(index: int, result: GenerateResult, *, multiple: bool) -> None:
    if multiple:
        print(f"-- prompt {index + 1} --")
    print(f"text: {result.text}")
    print(f"token_ids: {result.token_ids}")
    print(f"finish_reason: {result.finish_reason}")


async def run(args: argparse.Namespace) -> None:
    if args.num_prompts <= 0:
        raise ValueError("num_prompts must be positive")
    if args.long_prefill_token_threshold != 128:
        raise ValueError("--long-prefill-token-threshold must be 128 for DeepSeek V4")
    if args.stream and args.num_prompts != 1:
        raise ValueError("--stream currently requires --num-prompts 1")
    if args.enable_mtp and args.temperature > 0.0:
        raise ValueError("DeepSeek V4 MTP offline generation currently requires temperature=0")

    profile_config = create_profile_config(
        enabled=args.profile,
        output=Path(args.profile_output).expanduser().resolve(),
        levels=args.profile_level,
    )
    configure_profiler(
        profile_config,
        process_name="deepseek-v4-offline",
        initially_active=False,
    )
    engine_config = build_deepseek_v4_offline_engine_config(
        args.model_dir,
        device_ids=parse_device_ids(args.devices),
        model_id=args.model_id,
        platform=args.platform,
        max_seq_len=args.max_seq_len,
        max_new_tokens=args.max_new_tokens,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=args.long_prefill_token_threshold,
        npu_memory_utilization=args.npu_memory_utilization,
        weight_dtype=args.dtype,
        kv_dtype=args.kv_cache_dtype,
        enable_mtp=args.enable_mtp,
        enable_chunked_prefill=args.enable_chunked_prefill,
        kernel_cache_dir=args.kernel_cache_dir,
        save_kernels_dir=args.save_kernels_dir,
        pypto_root=args.pypto_root,
        profile_config=profile_config,
    )
    tokenizer = load_tokenizer(engine_config.model_dir)
    engine = AsyncLLMEngine(config=engine_config, tokenizer=tokenizer)
    generate_config = GenerateConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        stop=tuple(args.stop),
        stream=args.stream,
        ignore_eos=args.ignore_eos,
    )

    started = False
    profiling = False
    event_count = 0
    try:
        await engine.start()
        started = True
        if args.profile:
            start_profile()
            try:
                await engine.start_profile()
            except Exception:
                stop_profile()
                raise
            profiling = True

        start_time = time.perf_counter()
        if args.stream:
            request_id = engine.generate_request_id()
            previous_text = ""
            final_output = None
            async for output in engine.add_request(request_id, args.prompt, generate_config):
                delta = output.text[len(previous_text) :]
                if delta:
                    print(delta, end="", flush=True)
                previous_text = output.text
                if output.finished:
                    final_output = output
            print()
            if final_output is None:
                raise RuntimeError("Streaming generation ended without a final output")
            results = [
                GenerateResult(
                    text=final_output.text,
                    token_ids=list(final_output.token_ids),
                    finish_reason=engine.normalize_finish_reason(final_output.finish_reason),
                )
            ]
            print(f"token_ids: {results[0].token_ids}")
            print(f"finish_reason: {results[0].finish_reason}")
        else:
            results = await engine.generate_batch(
                [args.prompt] * args.num_prompts,
                generate_config,
            )
            for index, result in enumerate(results):
                _print_result(index, result, multiple=len(results) > 1)
        elapsed = time.perf_counter() - start_time
        num_tokens = sum(len(result.token_ids) for result in results)
        throughput = num_tokens / elapsed if elapsed > 0 else 0.0
        print(
            f"[perf] generated {num_tokens} tokens in {elapsed:.3f}s "
            f"-> {throughput:.2f} tok/s (overall, including prefill)"
        )
    finally:
        try:
            if profiling:
                try:
                    await engine.stop_profile()
                finally:
                    stop_profile()
                    event_count = merge_profile()
        finally:
            if started:
                await engine.stop()
            if args.profile:
                print(f"[profile] merged {event_count} events into {profile_config.trace_file}")


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for logger_name in ("simpler_setup", "pypto", "simpler"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
