# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Collapse Simpler's per-invocation trace into one host-clock lane per NPU rank."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROCESS_NAME_RE = re.compile(r"inv=(?P<inv>\d+) \(pid=(?P<pid>\d+)\)")
DEVICE_READY_RE = re.compile(r"\[chip_process pid=(?P<pid>\d+) dev=(?P<device>\d+)\] ready")


def callable_label(invocation: int) -> tuple[str, int | None, str]:
    prefill = {
        1: ("prefill.main", None, "rail_response"),
        2: ("prefill.main.lm_head", None, "rail_animation"),
        3: ("prefill.mtp", None, "rail_load"),
        4: ("prefill.mtp.lm_head", None, "rail_idle"),
    }
    if invocation in prefill:
        return prefill[invocation]
    step = (invocation - 5) // 4 + 1
    phase = (invocation - 5) % 4
    decode = {
        0: ("decode.main", "good"),
        1: ("decode.main.lm_head", "rail_animation"),
        2: ("decode.mtp", "cq_build_running"),
        3: ("decode.mtp.lm_head", "rail_idle"),
    }
    label, color = decode[phase]
    return label, step, color


def one_event(events: list[dict], name: str) -> dict | None:
    matches = [event for event in events if event.get("ph") == "X" and event.get("name") == name]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"expected one {name!r} event, found {len(matches)}")
    return matches[0]


def interval_union_us(events: list[dict], suffixes: tuple[str, ...]) -> float:
    intervals = sorted(
        (float(event["ts"]), float(event["ts"]) + float(event["dur"]))
        for event in events
        if event.get("ph") == "X" and str(event.get("name", "")).endswith(suffixes)
    )
    merged: list[list[float]] = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Simpler strace_timing Chrome trace")
    parser.add_argument("server_log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Draw host spans and visually project device-clock spans inside runner_run.",
    )
    parser.add_argument(
        "--host-only",
        action="store_true",
        help="With --detailed, omit device-clock projections.",
    )
    args = parser.parse_args()
    if args.host_only and not args.detailed:
        parser.error("--host-only requires --detailed")
    return args


def main() -> None:
    args = parse_args()
    source = json.loads(args.input.read_text())
    source_events = source["traceEvents"] if isinstance(source, dict) else source

    pid_to_device = {
        int(match.group("pid")): int(match.group("device"))
        for match in DEVICE_READY_RE.finditer(args.server_log.read_text(errors="replace"))
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

    roots = [
        event
        for events in grouped.values()
        for event in events
        if event.get("name") == "simpler_run"
    ]
    if not roots:
        raise ValueError("no simpler_run roots found")
    origin_us = min(float(event["ts"]) for event in roots)

    process_id = 1
    output_events: list[dict] = [
        {
            "ph": "M",
            "name": "process_name",
            "pid": process_id,
            "args": {"name": "DeepSeek V4 STRACE (8 NPU lanes)"},
        }
    ]
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
        device_wall = one_event(events, "simpler_run.runner_run.device_wall")
        label, step, color = callable_label(invocation)
        event_name = label if step is None else f"D{step:02d} {label}"
        has_device_trace = device_wall is not None
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
            "device_wall_ms": round(float(device_wall["dur"]) / 1000.0, 6) if device_wall else None,
            "effective_ms": (
                round(interval_union_us(events, (".orch", ".sched")) / 1000.0, 6)
                if has_device_trace
                else None
            ),
        }
        if not args.detailed:
            output_events.append(
                {
                    "ph": "X",
                    "name": event_name,
                    "cat": "simpler_run",
                    "pid": process_id,
                    "tid": device,
                    "ts": float(root["ts"]) - origin_us,
                    "dur": float(root["dur"]),
                    "cname": color,
                    "args": event_args,
                }
            )
            continue

        if runner is None:
            raise ValueError(f"invocation {invocation} has no runner_run span")
        if device_wall is None:
            if args.host_only:
                projected_device_start = 0.0
            else:
                raise ValueError(
                    f"invocation {invocation} has no device_wall span; use --host-only "
                    "when SIMPLER_DEVICE_STRACE_ENABLE=0"
                )
        else:
            projected_device_start = (
                float(runner["ts"]) + float(runner["dur"]) - float(device_wall["dur"])
            )
        for source_event in events:
            if source_event.get("ph") != "X":
                continue
            source_name = str(source_event.get("name", ""))
            is_device = (
                source_event.get("tid") == 1
                or source_event.get("args", {}).get("attrs") == "clk=dev"
            )
            if is_device:
                if args.host_only:
                    continue
                output_ts = projected_device_start + float(source_event["ts"]) - origin_us
                stage = source_name.removeprefix("simpler_run.runner_run.device_wall")
                stage = "device.wall" if not stage else f"device{stage}"
                clock_note = "visual projection: device wall right-aligned to host runner_run end"
                category = "strace.device.projected"
                stage_color = "thread_state_iowait"
            else:
                output_ts = float(source_event["ts"]) - origin_us
                stage = source_name.removeprefix("simpler_run")
                stage = "simpler_run" if not stage else stage.removeprefix(".")
                clock_note = "host CLOCK_MONOTONIC"
                category = "strace.host"
                stage_color = color if stage == "simpler_run" else "thread_state_running"
            output_events.append(
                {
                    "ph": "X",
                    "name": f"{event_name} | {stage}",
                    "cat": category,
                    "pid": process_id,
                    "tid": device,
                    "ts": output_ts,
                    "dur": float(source_event["dur"]),
                    "cname": stage_color,
                    "args": {
                        **event_args,
                        "strace_name": source_name,
                        "clock": clock_note,
                        "strace_depth": source_event.get("args", {}).get("depth"),
                    },
                }
            )

    output_events.sort(
        key=lambda event: (
            0 if event.get("ph") == "M" else 1,
            int(event.get("tid", 0)),
            float(event.get("ts", 0)),
        )
    )
    description = "Exactly eight NPU tracks. Collapsed simpler_run bars use host CLOCK_MONOTONIC."
    if args.detailed and args.host_only:
        description = "Exactly eight NPU tracks with authoritative host CLOCK_MONOTONIC spans only."
    elif args.detailed:
        description = (
            "Exactly eight NPU tracks. Host spans use CLOCK_MONOTONIC; device-clock spans are "
            "visual projections right-aligned to runner_run end and are not cross-rank timestamps."
        )
    payload = {
        "displayTimeUnit": "ms",
        "traceEvents": output_events,
        "metadata": {"description": description, "source": str(args.input)},
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    slice_count = sum(event.get("ph") == "X" for event in output_events)
    print(f"wrote {args.output}: {slice_count} slices from {invocation_count} invocations")


if __name__ == "__main__":
    main()
