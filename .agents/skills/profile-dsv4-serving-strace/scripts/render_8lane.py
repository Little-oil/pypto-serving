# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Merge the supported one-L2 host STRACE with the serving trace."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROCESS_NAME_RE = re.compile(r"inv=(?P<inv>\d+) \(pid=(?P<pid>\d+)\)")
DEVICE_READY_RE = re.compile(r"\[chip_process pid=(?P<pid>\d+) dev=(?P<device>\d+)\] ready")
MTP_ACCEPTANCE_RE = re.compile(r"MTP acceptance for .* proposed=(?P<steps>\d+)")


def callable_label(invocation: int) -> tuple[str, int | None, str]:
    prefill = {
        1: ("prefill.main", None, "rail_response"),
        2: ("prefill.main.lm_head", None, "rail_animation"),
        3: ("prefill.mtp", None, "rail_load"),
        4: ("prefill.mtp.lm_head", None, "rail_idle"),
    }
    if invocation in prefill:
        return prefill[invocation]
    return "decode.main+verify+mtp", invocation - 4, "good"


def one_event(events: list[dict], name: str) -> dict | None:
    matches = [event for event in events if event.get("ph") == "X" and event.get("name") == name]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected one {name!r} event, found {len(matches)}")
    return matches[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Simpler strace_timing Chrome trace")
    parser.add_argument("server_log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--serving-trace",
        type=Path,
        required=True,
        help="Serving SA_PROFILE trace on the shared host monotonic clock.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text())
    source_events = source["traceEvents"] if isinstance(source, dict) else source
    if any(
        event.get("ph") == "X"
        and (
            event.get("tid") == 1
            or event.get("args", {}).get("attrs") == "clk=dev"
        )
        for event in source_events
    ):
        raise ValueError("device STRACE is unsupported; expected host-only STRACE")

    server_log = args.server_log.read_text(errors="replace")
    pid_to_device = {
        int(match.group("pid")): int(match.group("device"))
        for match in DEVICE_READY_RE.finditer(server_log)
    }
    devices = sorted(pid_to_device.values())
    if len(devices) != 8 or len(set(devices)) != 8:
        raise ValueError(f"expected eight unique devices, got {pid_to_device}")

    virtual_processes: dict[int, tuple[int, int]] = {}
    for event in source_events:
        if event.get("ph") != "M" or event.get("name") != "process_name":
            continue
        match = PROCESS_NAME_RE.search(str(event.get("args", {}).get("name", "")))
        if match is None:
            continue
        raw_pid = int(match.group("pid"))
        if raw_pid in pid_to_device:
            virtual_processes[int(event["pid"])] = (
                pid_to_device[raw_pid],
                int(match.group("inv")),
            )

    grouped: dict[int, list[dict]] = {}
    for event in source_events:
        virtual_pid = event.get("pid")
        if virtual_pid in virtual_processes and event.get("ph") == "X":
            grouped.setdefault(int(virtual_pid), []).append(event)

    acceptance_matches = list(MTP_ACCEPTANCE_RE.finditer(server_log))
    if not acceptance_matches:
        raise ValueError("missing final MTP acceptance line")
    proposed_steps = int(acceptance_matches[-1].group("steps"))
    invocation_ids = sorted({invocation for _device, invocation in virtual_processes.values()})
    expected_invocation_ids = list(range(1, 5 + proposed_steps))
    if invocation_ids != expected_invocation_ids:
        raise ValueError(
            "expected four prefill invocations plus one fused invocation per decode step: "
            f"got={invocation_ids}, proposed={proposed_steps}"
        )

    roots = [
        event
        for events in grouped.values()
        for event in events
        if event.get("name") == "simpler_run"
    ]
    if not roots:
        raise ValueError("no simpler_run roots found")

    serving_source = json.loads(args.serving_trace.read_text())
    serving_events = (
        serving_source["traceEvents"]
        if isinstance(serving_source, dict)
        else serving_source
    )
    serving_timed_events = [event for event in serving_events if "ts" in event]
    serving_spans = [event for event in serving_events if event.get("ph") == "X"]
    if not serving_spans:
        raise ValueError(f"serving trace has no complete spans: {args.serving_trace}")
    host_start = min(float(event["ts"]) for event in roots)
    host_end = max(float(event["ts"]) + float(event["dur"]) for event in roots)
    serving_start = min(float(event["ts"]) for event in serving_spans)
    serving_end = max(
        float(event["ts"]) + float(event.get("dur", 0)) for event in serving_spans
    )
    if max(host_start, serving_start) >= min(host_end, serving_end):
        raise ValueError(
            "serving trace and host STRACE do not overlap on the host monotonic clock"
        )

    origin_us = min(float(event["ts"]) for event in roots + serving_timed_events)
    serving_pids = {
        int(event["pid"])
        for event in serving_events
        if isinstance(event.get("pid"), int)
    }
    process_id = max(serving_pids, default=0) + 1
    output_events: list[dict] = []
    for event in serving_events:
        output_event = dict(event)
        if "ts" in output_event:
            output_event["ts"] = float(output_event["ts"]) - origin_us
        output_events.append(output_event)
    output_events.extend(
        [
            {
                "ph": "M",
                "name": "process_name",
                "pid": process_id,
                "args": {"name": "Simpler host STRACE (8 NPU lanes)"},
            },
            {
                "ph": "M",
                "name": "process_sort_index",
                "pid": process_id,
                "args": {"sort_index": len(serving_pids)},
            },
        ]
    )
    for sort_index, device in enumerate(devices):
        output_events.extend(
            [
                {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": process_id,
                    "tid": device,
                    "args": {"name": f"device {device}"},
                },
                {
                    "ph": "M",
                    "name": "thread_sort_index",
                    "pid": process_id,
                    "tid": device,
                    "args": {"sort_index": sort_index},
                },
            ]
        )

    invocation_count = 0
    for virtual_pid, events in grouped.items():
        device, invocation = virtual_processes[virtual_pid]
        root = one_event(events, "simpler_run")
        if root is None:
            continue
        invocation_count += 1
        bind = one_event(events, "simpler_run.bind")
        runner = one_event(events, "simpler_run.runner_run")
        validate = one_event(events, "simpler_run.validate")
        label, step, color = callable_label(invocation)
        event_name = label if step is None else f"D{step:02d} {label}"
        event_args = {
            "device": device,
            "invocation": invocation,
            "callable": label,
            "decode_step": step,
            "hid": str(root.get("args", {}).get("hid", "")),
            "host_ms": round(float(root["dur"]) / 1000.0, 6),
            "bind_ms": round(float(bind["dur"]) / 1000.0, 6) if bind else None,
            "runner_run_ms": round(float(runner["dur"]) / 1000.0, 6) if runner else None,
            "validate_ms": round(float(validate["dur"]) / 1000.0, 6) if validate else None,
        }
        for source_event in events:
            if source_event.get("ph") != "X":
                continue
            source_name = str(source_event.get("name", ""))
            stage = source_name.removeprefix("simpler_run")
            stage = "simpler_run" if not stage else stage.removeprefix(".")
            output_events.append(
                {
                    "ph": "X",
                    "name": f"{event_name} | {stage}",
                    "cat": "strace.host",
                    "pid": process_id,
                    "tid": device,
                    "ts": float(source_event["ts"]) - origin_us,
                    "dur": float(source_event["dur"]),
                    "cname": color if stage == "simpler_run" else "thread_state_running",
                    "args": {
                        **event_args,
                        "strace_name": source_name,
                        "clock": "host CLOCK_MONOTONIC",
                        "strace_depth": source_event.get("args", {}).get("depth"),
                    },
                }
            )

    def sort_number(value: object) -> float:
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return float("inf")

    output_events.sort(
        key=lambda event: (
            0 if event.get("ph") == "M" else 1,
            sort_number(event.get("pid", 0)),
            sort_number(event.get("tid", 0)),
            float(event.get("ts", 0)),
        )
    )
    description = (
        "Serving SA_PROFILE spans and eight detailed one-L2 Simpler host STRACE lanes "
        "aligned on their shared host monotonic clock."
    )
    payload = {
        "displayTimeUnit": "ms",
        "traceEvents": output_events,
        "metadata": {
            "description": description,
            "sources": [str(args.serving_trace), str(args.input)],
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    slice_count = sum(event.get("ph") == "X" for event in output_events)
    print(f"wrote {args.output}: {slice_count} slices from {invocation_count} invocations")


if __name__ == "__main__":
    main()
