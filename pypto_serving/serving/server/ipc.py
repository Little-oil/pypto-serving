# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Lightweight IPC protocol for the engine↔worker boundary.

Replaces the old pickle-serialised full-graph path with
msgspec structs that ship only per-step deltas.  Prompt tokens and sampling
parameters are registered once per request (``NewRequestData``) and cached
inside the worker, so steady-state decode steps carry only:
  - per-request: last token, prev token, seq_len, block_ids
  - ~1 KB total instead of ~160 KB with the old full-graph pickle

Wire format: msgpack (msgspec), single encoded bytes object placed on the
existing ``multiprocessing.Queue``.  Switching the queue to a raw ``Pipe``
(Tier 2) is a drop-in swap at the ``encode_command`` / ``decode_command``
call sites.
"""

from __future__ import annotations

from typing import Union

import msgspec


# ---------------------------------------------------------------------------
# Request-scoped structs (engine → worker)
# ---------------------------------------------------------------------------

class NewRequestData(msgspec.Struct):
    """Full request data sent exactly once when a request is first scheduled.

    The worker caches this and references it by ``request_id`` for the lifetime
    of the request.  ``prompt_token_ids`` are never re-sent after this message.
    """

    request_id: str
    prompt_token_ids: list[int]
    temperature: float
    top_p: float
    top_k: int | None


class PrefillRequest(msgspec.Struct):
    """Per-request payload for a prefill (or chunked-prefill) step."""

    request_id: str
    # Token chunk to compute this step: prompt_token_ids[num_computed : num_computed+num_new]
    chunk_tokens: list[int]
    # Absolute position of the first token in this chunk (for RoPE).
    num_computed_tokens: int
    # KV-cache block table for this request (may grow step-over-step).
    block_ids: list[int]


class DecodeRequest(msgspec.Struct):
    """Per-request payload for a decode step — delta only, no prompt tokens."""

    request_id: str
    # output_token_ids[-1] (the token to decode from)
    last_token: int
    # output_token_ids[-2] if available, else prompt_token_ids[-1]  (for MTP prev context)
    prev_token: int
    # Total tokens computed so far: num_prompt_tokens + len(output_token_ids)
    seq_len: int
    # Full KV block table for this request.
    block_ids: list[int]


# ---------------------------------------------------------------------------
# Step-level commands (engine → worker)
# ---------------------------------------------------------------------------

class StepCommand(msgspec.Struct, tag="step"):
    """One scheduling step: admits new requests and runs prefill + decode.

    ``new_requests`` is non-empty only when freshly admitted requests enter the
    running state.  During steady-state decode it is always empty, keeping the
    payload at ~1 KB regardless of prompt length or batch size.
    """

    new_requests: list[NewRequestData]
    prefill_requests: list[PrefillRequest]
    decode_requests: list[DecodeRequest]
    # Request IDs that finished last step; worker releases device resources.
    finished_request_ids: list[str]


class ShutdownCommand(msgspec.Struct, tag="shutdown"):
    """Signals the worker to exit its busy-loop cleanly."""


# Union used for the decoder — tag field ("type") discriminates.
Command = Union[StepCommand, ShutdownCommand]


# ---------------------------------------------------------------------------
# Step result (worker → engine)
# ---------------------------------------------------------------------------

class StepResult(msgspec.Struct):
    """Sampled tokens returned after executing one step.

    Values are always ``list[int]``:
    - Standard decode: ``[token_id]`` (single element).
    - MTP / speculative: ``[t0, t1, ...]`` (multiple accepted tokens).

    The engine merges this back into the ``dict[str, int | list[int]]`` that
    ``scheduler.update_from_output`` expects.
    """

    new_tokens: dict[str, list[int]]
    error: str | None = None


# ---------------------------------------------------------------------------
# Codec — thin wrappers so call sites are transport-agnostic
# ---------------------------------------------------------------------------

_cmd_encoder: msgspec.msgpack.Encoder = msgspec.msgpack.Encoder()
_cmd_decoder: msgspec.msgpack.Decoder = msgspec.msgpack.Decoder(Command)
_result_encoder: msgspec.msgpack.Encoder = msgspec.msgpack.Encoder()
_result_decoder: msgspec.msgpack.Decoder = msgspec.msgpack.Decoder(StepResult)


def encode_command(cmd: Command) -> bytes:
    """Encode a command to bytes for queue transport."""
    return _cmd_encoder.encode(cmd)


def decode_command(data: bytes) -> Command:
    """Decode bytes from the input queue into a typed command."""
    return _cmd_decoder.decode(data)


def encode_result(result: StepResult) -> bytes:
    """Encode a step result to bytes for queue transport."""
    return _result_encoder.encode(result)


def decode_result(data: bytes) -> StepResult:
    """Decode bytes from the output queue into a typed step result."""
    return _result_decoder.decode(data)
