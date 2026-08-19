# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Merge aligned GBS32 serving spans with eight Host STRACE lanes."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


PROCESS_NAME_RE = re.compile(r"inv=(?P<inv>\d+) \(pid=(?P<pid>\d+)\)")
DEVICE_READY_RE = re.compile(r"\[chip_process pid=(?P<pid>\d+) dev=(?P<device>\d+)\] ready")
COLORS = {
    "prefill.main": "rail_response",
    "prefill.main.lm_head": "rail_animation",
    "prefill.mtp": "rail_load",
    "prefill.mtp.lm_head": "rail_idle",
    "decode.main+verify+mtp": "good",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Host-only Simpler Chrome trace")
    parser.add_argument("server_log", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--serving-trace", type=Path, required=True)
    parser.add_argument("--profile-summary", type=Path, required=True)
    return parser.parse_args()


def one_event(events: list[dict], name: str) -> dict | None:
    matches = [event for event in events if event.get("ph") == "X" and event.get("name") == name]
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name!r} event, got {len(matches)}")
    return matches[0]


def sort_number(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("inf")


def main() -> None:
    args = parse_args()
    simpler_payload = json.loads(args.input.read_text(encoding="utf-8"))
    source_events = simpler_payload["traceEvents"]
    if any(
        event.get("ph") == "X"
        and (
            event.get("tid") == 1
            or "clk=dev" in str(event.get("args", {}).get("attrs", ""))
        )
        for event in source_events
    ):
        raise RuntimeError("combined view requires Host-only STRACE")

    server_log = args.server_log.read_text(encoding="utf-8", errors="replace")
    pid_to_device = {
        int(match.group("pid")): int(match.group("device"))
        for match in DEVICE_READY_RE.finditer(server_log)
    }
    devices = sorted(pid_to_device.values())
    if len(devices) != 8 or len(set(devices)) != 8:
        raise RuntimeError(f"expected eight unique devices, got {pid_to_device}")

    summary = json.loads(args.profile_summary.read_text(encoding="utf-8"))
    label_by_hid = summary.get("simpler", summary)["callable_labels"]
    virtual_processes: dict[int, tuple[int, int]] = {}
    for event in source_events:
        if event.get("ph") != "M" or event.get("name") != "process_name":
            continue
        match = PROCESS_NAME_RE.search(str(event.get("args", {}).get("name", "")))
        if match is None:
            continue
        raw_pid = int(match.group("pid"))
        if raw_pid in pid_to_device:
            virtual_processes[int(event["pid"])] = (raw_pid, int(match.group("inv")))

    grouped: dict[int, list[dict]] = defaultdict(list)
    for event in source_events:
        virtual_pid = event.get("pid")
        if virtual_pid in virtual_processes and event.get("ph") == "X":
            grouped[int(virtual_pid)].append(event)
    if not grouped:
        raise RuntimeError("no profiled Host STRACE invocation found")

    roots = [
        event
        for events in grouped.values()
        for event in events
        if event.get("name") == "simpler_run"
    ]
    serving_payload = json.loads(args.serving_trace.read_text(encoding="utf-8"))
    serving_events = serving_payload["traceEvents"]
    serving_spans = [event for event in serving_events if event.get("ph") == "X"]
    serving_timed = [event for event in serving_events if "ts" in event]
    if not roots or not serving_spans:
        raise RuntimeError("Host or serving trace contains no complete span")

    host_start = min(float(event["ts"]) for event in roots)
    host_end = max(float(event["ts"]) + float(event["dur"]) for event in roots)
    serving_start = min(float(event["ts"]) for event in serving_spans)
    serving_end = max(
        float(event["ts"]) + float(event.get("dur", 0)) for event in serving_spans
    )
    if max(host_start, serving_start) >= min(host_end, serving_end):
        raise RuntimeError("serving and Host STRACE do not overlap on the shared host clock")

    origin_us = min(float(event["ts"]) for event in roots + serving_timed)
    serving_pids = {
        int(event["pid"])
        for event in serving_events
        if isinstance(event.get("pid"), int)
    }
    host_process_id = max(serving_pids, default=0) + 1
    output_events: list[dict] = []
    for event in serving_events:
        copied = dict(event)
        if "ts" in copied:
            copied["ts"] = float(copied["ts"]) - origin_us
        output_events.append(copied)
    output_events.extend(
        [
            {
                "ph": "M",
                "name": "process_name",
                "pid": host_process_id,
                "args": {"name": "Simpler host STRACE (8 NPU lanes, aligned GBS32)"},
            },
            {
                "ph": "M",
                "name": "process_sort_index",
                "pid": host_process_id,
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
                    "pid": host_process_id,
                    "tid": device,
                    "args": {"name": f"device {device}"},
                },
                {
                    "ph": "M",
                    "name": "thread_sort_index",
                    "pid": host_process_id,
                    "tid": device,
                    "args": {"sort_index": sort_index},
                },
            ]
        )

    sequence_by_pid: dict[int, Counter] = defaultdict(Counter)
    invocation_count = 0
    for virtual_pid, events in sorted(
        grouped.items(), key=lambda item: float(one_event(item[1], "simpler_run")["ts"])
    ):
        root = one_event(events, "simpler_run")
        if root is None:
            continue
        raw_pid, raw_invocation = virtual_processes[virtual_pid]
        device = pid_to_device[raw_pid]
        hid = str(root.get("args", {}).get("hid", ""))
        label = label_by_hid.get(hid)
        if label is None:
            raise RuntimeError(f"unclassified callable hid={hid}")
        sequence_by_pid[raw_pid][label] += 1
        sequence = sequence_by_pid[raw_pid][label]
        decode_step = sequence if label.startswith("decode") else None
        event_name = (
            f"D{sequence:02d} {label}"
            if decode_step is not None
            else f"P{sequence:03d} {label}"
        )
        bind = one_event(events, "simpler_run.bind")
        runner = one_event(events, "simpler_run.runner_run")
        validate = one_event(events, "simpler_run.validate")
        common_args = {
            "device": device,
            "invocation": raw_invocation,
            "callable": label,
            "decode_step": decode_step,
            "hid": hid,
            "host_ms": round(float(root["dur"]) / 1000.0, 6),
            "bind_ms": round(float(bind["dur"]) / 1000.0, 6) if bind else None,
            "runner_run_ms": round(float(runner["dur"]) / 1000.0, 6) if runner else None,
            "validate_ms": round(float(validate["dur"]) / 1000.0, 6) if validate else None,
        }
        invocation_count += 1
        for source_event in events:
            if source_event.get("ph") != "X":
                continue
            source_name = str(source_event.get("name", ""))
            stage = source_name.removeprefix("simpler_run").removeprefix(".")
            stage = stage or "simpler_run"
            output_events.append(
                {
                    "ph": "X",
                    "name": f"{event_name} | {stage}",
                    "cat": "strace.host",
                    "pid": host_process_id,
                    "tid": device,
                    "ts": float(source_event["ts"]) - origin_us,
                    "dur": float(source_event["dur"]),
                    "cname": COLORS[label] if stage == "simpler_run" else "thread_state_running",
                    "args": {
                        **common_args,
                        "strace_name": source_name,
                        "clock": "host CLOCK_MONOTONIC",
                        "strace_depth": source_event.get("args", {}).get("depth"),
                    },
                }
            )

    output_events.sort(
        key=lambda event: (
            0 if event.get("ph") == "M" else 1,
            sort_number(event.get("pid", 0)),
            sort_number(event.get("tid", 0)),
            float(event.get("ts", 0)),
        )
    )
    payload = {
        "displayTimeUnit": "ms",
        "traceEvents": output_events,
        "metadata": {
            "description": (
                "Aligned GBS32 serving SA_PROFILE spans and eight detailed one-L2 Simpler "
                "Host STRACE lanes on their shared host monotonic clock."
            ),
            "sources": [str(args.serving_trace), str(args.input)],
            "workload": "GBS32/DP8/per-rank-batch4/Seq64/Output256/MTP1/temperature0",
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    categories = Counter(
        str(event.get("cat", ""))
        for event in output_events
        if event.get("ph") == "X"
    )
    print(
        f"wrote {args.output}: invocations={invocation_count} "
        f"categories={dict(sorted(categories.items()))}"
    )


if __name__ == "__main__":
    main()
