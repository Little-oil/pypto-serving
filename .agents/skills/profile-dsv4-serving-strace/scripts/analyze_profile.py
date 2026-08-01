# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Build Simpler host swimlanes and serving-span summaries from one DSV4 run."""

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
EFFECTIVE_US_INDEX = 2

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

    split_log = server_log.replace("[STRACE]", "\n[STRACE]")
    spans = list(parse_spans(split_log.splitlines()))
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

    metric_by_key = {
        (invocation.pid, invocation.inv, invocation.hid): _round_metrics(invocation)
        for invocation in request_invocations
    }
    device_trace_invocations = [
        invocation
        for invocation in request_invocations
        if any(span.is_device for span in invocation.spans)
    ]
    if device_trace_invocations and len(device_trace_invocations) != len(request_invocations):
        raise RuntimeError(
            "partial device STRACE data: "
            f"{len(device_trace_invocations)}/{len(request_invocations)} invocations"
        )
    device_effective_available = bool(device_trace_invocations)
    hid_by_inv: dict[int, str] = {}
    for invocation in request_invocations:
        hid_by_inv.setdefault(invocation.inv, invocation.hid)

    invocation_ids = sorted(hid_by_inv)
    if (
        invocation_ids != list(range(1, invocation_ids[-1] + 1))
        or invocation_ids[-1] < 8
        or (invocation_ids[-1] - 4) % 4
    ):
        raise RuntimeError(f"unexpected invocation ids: {invocation_ids}")
    decode_step_count = (invocation_ids[-1] - 4) // 4
    if len(request_invocations) != len(devices) * invocation_ids[-1]:
        raise RuntimeError(
            f"incomplete rank data: got {len(request_invocations)} invocations, "
            f"expected {len(devices) * invocation_ids[-1]}"
        )

    pids = sorted(pid_to_device, key=pid_to_device.get)

    def metric_us(pid: int, invocation_ids_for_phase: list[int], metric_index: int) -> float:
        return sum(
            metric_by_key[(pid, invocation, hid_by_inv[invocation])][metric_index]
            for invocation in invocation_ids_for_phase
        )

    def rank_phase_row(pid: int, main_ids: list[int], mtp_ids: list[int]) -> dict:
        main_host = metric_us(pid, main_ids, HOST_US_INDEX)
        mtp_host = metric_us(pid, mtp_ids, HOST_US_INDEX)
        row = {
            "device": pid_to_device[pid],
            "main_host_us": main_host,
            "mtp_host_us": mtp_host,
            "total_host_us": main_host + mtp_host,
        }
        if device_effective_available:
            main_effective = metric_us(pid, main_ids, EFFECTIVE_US_INDEX)
            mtp_effective = metric_us(pid, mtp_ids, EFFECTIVE_US_INDEX)
            row.update(
                {
                    "main_effective_us": main_effective,
                    "mtp_effective_us": mtp_effective,
                    "total_effective_us": main_effective + mtp_effective,
                }
            )
        return row

    prefill_rows = []
    for pid in pids:
        prefill_rows.append(rank_phase_row(pid, [1, 2], [3, 4]))

    decode_rows = []
    critical_metric = "total_effective_us" if device_effective_available else "total_host_us"
    for step in range(decode_step_count):
        base = 5 + step * 4
        per_rank = []
        for pid in pids:
            per_rank.append(rank_phase_row(pid, [base, base + 1], [base + 2, base + 3]))
        critical = max(per_rank, key=lambda row: row[critical_metric])
        decode_row = {
            "step": step + 1,
            "critical_device": critical["device"],
            "critical_main_host_us": critical["main_host_us"],
            "critical_mtp_host_us": critical["mtp_host_us"],
            "critical_total_host_us": critical["total_host_us"],
            "per_rank": per_rank,
        }
        if device_effective_available:
            decode_row.update(
                {
                    "critical_main_effective_us": critical["main_effective_us"],
                    "critical_mtp_effective_us": critical["mtp_effective_us"],
                    "critical_total_effective_us": critical["total_effective_us"],
                }
            )
        decode_rows.append(decode_row)

    serving_events = serving_trace["traceEvents"]
    kernel_durations_ms: dict[str, list[float]] = defaultdict(list)
    for event in serving_events:
        if (
            event.get("ph") == "X"
            and event.get("cat") == "kernel"
            and not event.get("name", "").endswith(".worker_run")
        ):
            kernel_durations_ms[event["args"]["kernel"]].append(event["dur"] / 1000.0)

    required_kernels = {
        "deepseek_v4_prefill",
        "deepseek_v4_mtp_prefill",
        "deepseek_v4_decode",
        "deepseek_v4_mtp_decode",
    }
    if not required_kernels.issubset(kernel_durations_ms):
        raise RuntimeError(f"missing serving kernel spans: {required_kernels - kernel_durations_ms.keys()}")

    request_ms = next(
        (
            event["dur"] / 1000.0
            for event in serving_events
            if event.get("ph") == "X" and event.get("name") == "http.completions"
        ),
        None,
    )
    if request_ms is None:
        raise RuntimeError("serving trace has no 'http.completions' span")
    completion_match = re.search(
        r"request .* finished: prompt=(\d+) out=(\d+).*e2e=([0-9.]+)s",
        server_log,
    )
    acceptance_match = re.search(
        r"MTP acceptance for .* accepted=(\d+) proposed=(\d+) rate=([0-9.]+)%",
        server_log,
    )
    if completion_match is None or acceptance_match is None:
        raise RuntimeError("missing request completion or final MTP acceptance line")
    completion_tokens = int(completion_match.group(2))
    if completion_tokens != args.expected_tokens:
        raise RuntimeError(
            f"expected {args.expected_tokens} completion tokens, got {completion_tokens}"
        )

    steady = decode_rows[1:]
    if not steady:
        raise RuntimeError("need at least two decode iterations for steady statistics")

    def kernel_summary(name: str) -> dict:
        samples = kernel_durations_ms[name]
        return {
            "samples": samples,
            "all": stats(samples),
            "steady_after_first": stats(samples[1:]) if len(samples) > 1 else None,
        }

    critical_fields = [
        "critical_main_host_us",
        "critical_mtp_host_us",
        "critical_total_host_us",
    ]
    if device_effective_available:
        critical_fields.extend(
            [
                "critical_main_effective_us",
                "critical_mtp_effective_us",
                "critical_total_effective_us",
            ]
        )
    serving_span_events = [event for event in serving_events if event.get("ph") == "X"]
    serving_categories = Counter(str(event.get("cat", "")) for event in serving_span_events)
    required_serving_categories = {
        "request",
        "serving",
        "scheduler",
        "worker",
        "executor",
        "kernel",
    }
    missing_categories = required_serving_categories - serving_categories.keys()
    if missing_categories:
        raise RuntimeError(f"missing serving profile categories: {sorted(missing_categories)}")

    summary = {
        "run_id": args.run_id,
        "devices": devices,
        "request": {
            "prompt_tokens": int(completion_match.group(1)),
            "completion_tokens": completion_tokens,
            "wall_ms": request_ms,
            "reported_e2e_s": float(completion_match.group(3)),
            "mtp_accepted": int(acceptance_match.group(1)),
            "mtp_proposed": int(acceptance_match.group(2)),
            "mtp_acceptance_percent": float(acceptance_match.group(3)),
        },
        "serving_kernel_ms": {
            name: kernel_summary(name)
            for name in sorted(required_kernels)
        },
        "serving_profile": {
            "event_count": len(serving_events),
            "span_count": len(serving_span_events),
            "span_categories": dict(sorted(serving_categories.items())),
        },
        "simpler": {
            "span_count": len(spans),
            "request_invocation_count": len(request_invocations),
            "device_effective_available": device_effective_available,
            "prefill_per_rank": prefill_rows,
            "prefill_critical": max(prefill_rows, key=lambda row: row[critical_metric]),
            "decode_steps": decode_rows,
            "decode_critical_all_us": {
                field: stats([row[field] for row in decode_rows])
                for field in critical_fields
            },
            "decode_critical_steady_us": {
                field: stats([row[field] for row in steady])
                for field in critical_fields
            },
        },
    }
    (artifact_dir / "profile-summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    prefill = summary["simpler"]["prefill_critical"]
    main_serving = summary["serving_kernel_ms"]["deepseek_v4_decode"]["steady_after_first"]
    mtp_serving = summary["serving_kernel_ms"]["deepseek_v4_mtp_decode"]["steady_after_first"]
    if main_serving is None or mtp_serving is None:
        raise RuntimeError("need more than one decode kernel span for steady statistics")
    critical_rank_counts = Counter(row["critical_device"] for row in decode_rows)
    critical_rank_summary = ", ".join(
        f"device {device}: {count}/{decode_step_count}"
        for device, count in sorted(critical_rank_counts.items())
    )
    host_decode_table = "\n".join(
        "| "
        f"{row['step']} | {row['critical_device']} | "
        f"{row['critical_main_host_us'] / 1000:.3f} | "
        f"{row['critical_mtp_host_us'] / 1000:.3f} | "
        f"{row['critical_total_host_us'] / 1000:.3f} |"
        for row in decode_rows
    )
    host_stats = summary["simpler"]["decode_critical_steady_us"]
    effective_section = """
## Simpler Device Effective

Device STRACE is disabled (`SIMPLER_DEVICE_STRACE_ENABLE=0`), so this run intentionally
does not contain `device_wall`, `orch`, `sched`, or Effective measurements.
"""
    if device_effective_available:
        steady_effective = summary["simpler"]["decode_critical_steady_us"]
        effective_decode_table = "\n".join(
            "| "
            f"{row['step']} | {row['critical_device']} | "
            f"{row['critical_main_effective_us'] / 1000:.3f} | "
            f"{row['critical_mtp_effective_us'] / 1000:.3f} | "
            f"{row['critical_total_effective_us'] / 1000:.3f} |"
            for row in decode_rows
        )
        effective_section = f"""
## Simpler Device Effective

This input contains device STRACE. Effective is the union of the device-domain Orch and
Sched windows.

- Prefill critical rank: device {prefill["device"]}, main={prefill["main_effective_us"] / 1000:.3f} ms, MTP={prefill["mtp_effective_us"] / 1000:.3f} ms, total={prefill["total_effective_us"] / 1000:.3f} ms
- Decode steady critical rank mean: main={steady_effective["critical_main_effective_us"]["mean"] / 1000:.3f} ms, MTP={steady_effective["critical_mtp_effective_us"]["mean"] / 1000:.3f} ms, total={steady_effective["critical_total_effective_us"]["mean"] / 1000:.3f} ms/iteration

| Decode iteration | Critical device | Main Effective (ms) | MTP Effective (ms) | Total Effective (ms) |
| ---: | ---: | ---: | ---: | ---: |
{effective_decode_table}
"""
    report = f"""# DeepSeek V4 serving profile: {completion_tokens} output tokens

- Run: `{args.run_id}`
- Devices: `{",".join(str(device) for device in devices)}`
- Request: prompt={summary["request"]["prompt_tokens"]}, output={completion_tokens}, HTTP wall={request_ms:.3f} ms
- MTP acceptance: {summary["request"]["mtp_accepted"]}/{summary["request"]["mtp_proposed"]} ({summary["request"]["mtp_acceptance_percent"]:.2f}%)
- Serving profile: {len(serving_span_events)} spans, categories={dict(sorted(serving_categories.items()))}

## Serving trace wall time

- Prefill main kernel span: {kernel_durations_ms["deepseek_v4_prefill"][0]:.3f} ms
- Prefill MTP kernel span: {kernel_durations_ms["deepseek_v4_mtp_prefill"][0]:.3f} ms
- Decode main steady mean (steps 2-{decode_step_count}): {main_serving["mean"]:.3f} ms/iteration
- Decode MTP steady mean (steps 2-{decode_step_count}): {mtp_serving["mean"]:.3f} ms/iteration

## Simpler Host STRACE

- Prefill critical rank: device {prefill["device"]}, main={prefill["main_host_us"] / 1000:.3f} ms, MTP={prefill["mtp_host_us"] / 1000:.3f} ms, total={prefill["total_host_us"] / 1000:.3f} ms
- Decode steady critical rank mean (steps 2-{decode_step_count}): main={host_stats["critical_main_host_us"]["mean"] / 1000:.3f} ms, MTP={host_stats["critical_mtp_host_us"]["mean"] / 1000:.3f} ms, total={host_stats["critical_total_host_us"]["mean"] / 1000:.3f} ms/iteration
- Critical-rank counts across decode: {critical_rank_summary}

| Decode iteration | Critical device | Main host (ms) | MTP host (ms) | Total host (ms) |
| ---: | ---: | ---: | ---: | ---: |
{host_decode_table}

{effective_section}

## Artifacts

- `completion.txt`: generated text
- `completion-response.json`: complete HTTP response
- `serving-trace/trace.json`: serving/framework `SA_PROFILE=verbose` swimlane
- `simpler-swimlane.json`: full Simpler host STRACE trace
- `strace-8lane.json`: exactly eight host-clock NPU lanes
- `server.log`: raw serving and Simpler log
- `profile-summary.json`: complete serving and per-step/per-rank numbers
"""
    (artifact_dir / "profile-summary.md").write_text(report)
    print(artifact_dir / "profile-summary.md")


if __name__ == "__main__":
    main()
