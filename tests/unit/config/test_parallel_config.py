# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from __future__ import annotations


import pytest

from pypto_serving.config.parallel import ParallelConfig, parse_device_ids


def test_parallel_config_groups_dp_replicas_into_tp_groups():
    config = ParallelConfig(
        data_parallel_size=2,
        tensor_parallel_size=2,
        devices=(0, 1, 2, 3),
    )

    assert config.replica_device_groups == ((0, 1), (2, 3))
    assert config.for_replica((2, 3)).data_parallel_size == 1
    assert config.for_replica((2, 3)).devices == (2, 3)


def test_parallel_config_rejects_unsupported_modes():
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        ParallelConfig(pipeline_parallel_size=2)

    with pytest.raises(ValueError, match="expert parallel"):
        ParallelConfig(enable_expert_parallel=True)

    with pytest.raises(ValueError, match="duplicates"):
        ParallelConfig(data_parallel_size=1, tensor_parallel_size=2, devices=(0, 0))


def test_parse_device_ids_uses_default_device():
    assert parse_device_ids(None, default_device=3) == (3,)
    assert parse_device_ids("0, 2,4", default_device=3) == (0, 2, 4)
