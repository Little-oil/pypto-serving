# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Analyze the single supported DSV4 fused one-L2 host-STRACE profile."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
HOST_US_INDEX = 0
DECODE_KERNEL = "deepseek_v4_decode_mtp_fused"
REQUIRED_KERNELS = {
    "deepseek_v4_prefill",
    "deepseek_v4_mtp_prefill",
    DECODE_KERNEL,
}

try:
    from simpler_setup.tools.strace_timing import (
        _round_metrics,
        bucket_by_hid,
        group_invocations,
        parse_spans,
        to_chrome_trace,
    )
except ModuleNotFoundError:
    runtime_candidates = [
        Path(value)
        for value in (
            os.environ.get("PYPTO_RUNTIME_ROOT"),
            str(REPO_ROOT.parent / "pypto" / "runtime"),
        )
        if value
    ]
    runtime_root = next(
        (candidate for candidate in runtime_candidates if (candidate / "simpler_setup").is_dir()),
        None,
    )
    if runtime_root is None:
        raise RuntimeError(
            "cannot import simpler_setup; install the PyPTO runtime package or set "
            "PYPTO_RUNTIME_ROOT to the runtime directory"
        ) from None
    sys.path.insert(0, str(runtime_root))
    from simpler_setup.tools.strace_timing import (  # noqa: E402
        _round_metrics,
        bucket_by_hid,
        group_invocations,
        parse_spans,
        to_chrome_trace,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--run-id", "--task-id", dest="run_id", default="unknown")
    parser.add_argument("--expected-tokens", type=int, default=20)
    return parser.parse_args()


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    server_log_path = artifact_dir / "server.log"
    serving_trace_path = artifact_dir / "serving-trace" / "trace.json"
    server_log = server_log_path.read_text(encoding="utf-8", errors="replace")
    serving_trace = json.loads(serving_trace_path.read_text(encoding="utf-8"))
    serving_events = serving_trace["traceEvents"]

    kernel_durations_ms: dict[str, list[float]] = defaultdict(list)
    for event in serving_events:
        if (
            event.get("ph") == "X"
            and event.get("cat") == "kernel"
            and not event.get("name", "").endswith(".worker_run")
        ):
            kernel_durations_ms[event["args"]["kernel"]].append(event["dur"] / 1000.0)
    missing_kernels = REQUIRED_KERNELS - kernel_durations_ms.keys()
    if missing_kernels:
        raise RuntimeError(f"missing one-L2 serving kernel spans: {sorted(missing_kernels)}")
    if "deepseek_v4_decode" in kernel_durations_ms or "deepseek_v4_mtp_decode" in kernel_durations_ms:
        raise RuntimeError("split decode kernels are unsupported; expected fused one-L2 decode")

    split_log = server_log.replace("[STRACE]", "\n[STRACE]")
    spans = list(parse_spans(split_log.splitlines()))
    if any(span.is_device for span in spans):
        raise RuntimeError("device STRACE is unsupported; set SIMPLER_DEVICE_STRACE_ENABLE=0")
    invocations = group_invocations(spans)
    pid_to_device = {
        int(pid): int(device)
        for pid, device in re.findall(r"\[chip_process pid=(\d+) dev=(\d+)\] ready", server_log)
    }
    devices = sorted(pid_to_device.values())
    if len(devices) != 8 or len(set(devices)) != 8:
        raise RuntimeError(f"expected eight unique device mappings, got {devices}")

    request_invocations = [
        invocation
        for invocation in invocations
        if invocation.inv > 0 and invocation.pid in pid_to_device
    ]
    if not request_invocations:
        raise RuntimeError("no request STRACE invocations found")
    simpler_trace = to_chrome_trace(request_invocations, bucket_by_hid(request_invocations))
    (artifact_dir / "simpler-swimlane.json").write_text(json.dumps(simpler_trace))

    acceptance_matches = list(
        re.finditer(
            r"MTP acceptance for .* accepted=(\d+) proposed=(\d+) rate=([0-9.]+)%",
            server_log,
        )
    )
    if not acceptance_matches:
        raise RuntimeError("missing final MTP acceptance line")
    acceptance_match = acceptance_matches[-1]
    proposed_steps = int(acceptance_match.group(2))

    hid_by_inv: dict[int, str] = {}
    for invocation in request_invocations:
        hid_by_inv.setdefault(invocation.inv, invocation.hid)
    invocation_ids = sorted(hid_by_inv)
    expected_invocation_ids = list(range(1, 5 + proposed_steps))
    if invocation_ids != expected_invocation_ids:
        raise RuntimeError(
            "expected four prefill invocations plus one fused invocation per decode step: "
            f"got={invocation_ids}, proposed={proposed_steps}"
        )
    if len(kernel_durations_ms[DECODE_KERNEL]) != proposed_steps:
        raise RuntimeError(
            f"expected {proposed_steps} fused decode kernel spans, "
            f"got {len(kernel_durations_ms[DECODE_KERNEL])}"
        )
    expected_rank_invocations = len(devices) * len(invocation_ids)
    if len(request_invocations) != expected_rank_invocations:
        raise RuntimeError(
            f"incomplete rank data: got {len(request_invocations)} invocations, "
            f"expected {expected_rank_invocations}"
        )

    metric_by_key = {
        (invocation.pid, invocation.inv, invocation.hid): _round_metrics(invocation)
        for invocation in request_invocations
    }
    pids = sorted(pid_to_device, key=pid_to_device.get)

    def metric_us(pid: int, ids: list[int]) -> float:
        return sum(
            metric_by_key[(pid, invocation_id, hid_by_inv[invocation_id])][HOST_US_INDEX]
            for invocation_id in ids
        )

    def prefill_row(pid: int) -> dict:
        main_host = metric_us(pid, [1, 2])
        mtp_host = metric_us(pid, [3, 4])
        return {
            "device": pid_to_device[pid],
            "main_host_us": main_host,
            "mtp_host_us": mtp_host,
            "total_host_us": main_host + mtp_host,
        }

    prefill_rows = [prefill_row(pid) for pid in pids]
    decode_rows = []
    for step in range(proposed_steps):
        invocation_id = 5 + step
        per_rank = []
        for pid in pids:
            host_us = metric_us(pid, [invocation_id])
            per_rank.append(
                {
                    "device": pid_to_device[pid],
                    "fused_host_us": host_us,
                    "total_host_us": host_us,
                }
            )
        critical = max(per_rank, key=lambda row: row["total_host_us"])
        decode_rows.append(
            {
                "step": step + 1,
                "critical_device": critical["device"],
                "critical_fused_host_us": critical["fused_host_us"],
                "critical_total_host_us": critical["total_host_us"],
                "per_rank": per_rank,
            }
        )

    request_ms = next(
        (
            event["dur"] / 1000.0
            for event in serving_events
            if event.get("ph") == "X" and event.get("name") == "http.completions"
        ),
        None,
    )
    completion_match = re.search(
        r"request .* finished: prompt=(\d+) out=(\d+).*e2e=([0-9.]+)s",
        server_log,
    )
    if request_ms is None or completion_match is None:
        raise RuntimeError("missing HTTP span or request completion line")
    completion_tokens = int(completion_match.group(2))
    if completion_tokens != args.expected_tokens:
        raise RuntimeError(
            f"expected {args.expected_tokens} completion tokens, got {completion_tokens}"
        )
    steady = decode_rows[1:]
    if not steady:
        raise RuntimeError("need at least two decode iterations for steady statistics")

    serving_spans = [event for event in serving_events if event.get("ph") == "X"]
    serving_categories = Counter(str(event.get("cat", "")) for event in serving_spans)
    required_categories = {"request", "serving", "scheduler", "worker", "executor", "kernel"}
    missing_categories = required_categories - serving_categories.keys()
    if missing_categories:
        raise RuntimeError(f"missing serving profile categories: {sorted(missing_categories)}")

    def kernel_summary(name: str) -> dict:
        samples = kernel_durations_ms[name]
        return {
            "samples": samples,
            "all": stats(samples),
            "steady_after_first": stats(samples[1:]) if len(samples) > 1 else None,
        }

    summary = {
        "run_id": args.run_id,
        "devices": devices,
        "decode_layout": "one_l2",
        "request": {
            "prompt_tokens": int(completion_match.group(1)),
            "completion_tokens": completion_tokens,
            "wall_ms": request_ms,
            "reported_e2e_s": float(completion_match.group(3)),
            "mtp_accepted": int(acceptance_match.group(1)),
            "mtp_proposed": proposed_steps,
            "mtp_acceptance_percent": float(acceptance_match.group(3)),
        },
        "serving_kernel_ms": {
            name: kernel_summary(name)
            for name in sorted(REQUIRED_KERNELS)
        },
        "serving_profile": {
            "event_count": len(serving_events),
            "span_count": len(serving_spans),
            "span_categories": dict(sorted(serving_categories.items())),
        },
        "simpler": {
            "span_count": len(spans),
            "request_invocation_count": len(request_invocations),
            "device_effective_available": False,
            "prefill_per_rank": prefill_rows,
            "prefill_critical": max(prefill_rows, key=lambda row: row["total_host_us"]),
            "decode_steps": decode_rows,
            "decode_critical_all_us": {
                "critical_fused_host_us": stats(
                    [row["critical_fused_host_us"] for row in decode_rows]
                ),
                "critical_total_host_us": stats(
                    [row["critical_total_host_us"] for row in decode_rows]
                ),
            },
            "decode_critical_steady_us": {
                "critical_fused_host_us": stats(
                    [row["critical_fused_host_us"] for row in steady]
                ),
                "critical_total_host_us": stats(
                    [row["critical_total_host_us"] for row in steady]
                ),
            },
        },
    }
    (artifact_dir / "profile-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    prefill = summary["simpler"]["prefill_critical"]
    fused_serving = summary["serving_kernel_ms"][DECODE_KERNEL]["steady_after_first"]
    if fused_serving is None:
        raise RuntimeError("need more than one fused decode kernel span")
    host_stats = summary["simpler"]["decode_critical_steady_us"]
    critical_rank_counts = Counter(row["critical_device"] for row in decode_rows)
    critical_rank_summary = ", ".join(
        f"device {device}: {count}/{proposed_steps}"
        for device, count in sorted(critical_rank_counts.items())
    )
    host_decode_table = "\n".join(
        f"| {row['step']} | {row['critical_device']} | "
        f"{row['critical_fused_host_us'] / 1000:.3f} |"
        for row in decode_rows
    )
    report = f"""# DeepSeek V4 serving profile: {completion_tokens} output tokens

- Run: `{args.run_id}`
- Devices: `{','.join(str(device) for device in devices)}`
- Layout: fused one-L2
- Request: prompt={summary['request']['prompt_tokens']}, output={completion_tokens}, HTTP wall={request_ms:.3f} ms
- MTP acceptance: {summary['request']['mtp_accepted']}/{proposed_steps} ({summary['request']['mtp_acceptance_percent']:.2f}%)
- Serving profile: {len(serving_spans)} spans, categories={dict(sorted(serving_categories.items()))}

## Serving trace wall time

- Prefill main kernel span: {kernel_durations_ms['deepseek_v4_prefill'][0]:.3f} ms
- Prefill MTP kernel span: {kernel_durations_ms['deepseek_v4_mtp_prefill'][0]:.3f} ms
- Fused decode steady mean (steps 2-{proposed_steps}): {fused_serving['mean']:.3f} ms/iteration

## Simpler Host STRACE

- Prefill critical rank: device {prefill['device']}, main={prefill['main_host_us'] / 1000:.3f} ms, MTP={prefill['mtp_host_us'] / 1000:.3f} ms, total={prefill['total_host_us'] / 1000:.3f} ms
- Decode steady critical rank mean: fused main+verify+MTP={host_stats['critical_fused_host_us']['mean'] / 1000:.3f} ms/iteration
- Critical-rank counts across decode: {critical_rank_summary}

| Decode iteration | Critical device | Fused main+verify+MTP host (ms) |
| ---: | ---: | ---: |
{host_decode_table}

## Simpler Device Effective

Device STRACE is disabled (`SIMPLER_DEVICE_STRACE_ENABLE=0`), so Effective is unavailable.

## Artifacts

- `completion.txt`: generated text
- `completion-response.json`: complete HTTP response
- `serving-trace/trace.json`: serving/framework `SA_PROFILE=verbose` swimlane
- `simpler-swimlane.json`: full Simpler host STRACE trace
- `serving-strace-swimlane.json`: serving spans and eight host STRACE lanes on one timeline
- `server.log`: raw serving and Simpler log
- `profile-summary.json`: complete serving and per-step/per-rank numbers
"""
    (artifact_dir / "profile-summary.md").write_text(report)
    print(artifact_dir / "profile-summary.md")


if __name__ == "__main__":
    main()
