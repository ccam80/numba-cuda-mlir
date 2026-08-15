# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for typed intra-block statement scheduling."""

import numpy as np
import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir._whole_function_planners import (
    _typed_planner_registry,
)
from numba_cuda_mlir.extending import (
    TypedWholeFunctionPlanner,
    WholeFunctionPlanner,
    register_typed_planner,
)
from numba_cuda_mlir.numba_cuda.core import ir
from numba_cuda_mlir.typed_block_scheduler import (
    TypedBlockScheduler,
    _METADATA_KEY,
)


@pytest.fixture
def isolated_typed_planners():
    with _typed_planner_registry._lock:
        original = list(_typed_planner_registry._planners)
        _typed_planner_registry._planners.clear()
    original_policy = TypedBlockScheduler.policy
    try:
        yield
    finally:
        with _typed_planner_registry._lock:
            _typed_planner_registry._planners[:] = original
        TypedBlockScheduler.policy = original_policy


def test_typed_registry_rejects_untyped_planners():
    class UntypedPlanner(WholeFunctionPlanner):
        def run(self):
            return False

    with pytest.raises(TypeError, match="TypedWholeFunctionPlanner subclass"):
        register_typed_planner(UntypedPlanner)


def test_unknown_policy_is_rejected(isolated_typed_planners):
    if not cuda.is_available():
        pytest.skip("CUDA GPU required")
    TypedBlockScheduler.policy = "not_a_policy"
    register_typed_planner(TypedBlockScheduler)

    @cuda.jit
    def kernel(out):
        out[0] = 1

    out = np.zeros(1, dtype=np.int32)
    with pytest.raises(Exception, match="unknown block schedule policy"):
        kernel[1, 1](out)


def _capture_bodies():
    """Planner that records every block body after scheduling ran."""

    captured = {}

    class CapturePlanner(TypedWholeFunctionPlanner):
        def run(self):
            for label, block in self.state.func_ir.blocks.items():
                captured[label] = list(block.body)
            captured["stats"] = self.state.metadata.get(_METADATA_KEY)
            return False

    return CapturePlanner, captured


def _run_scheduled_kernel(policy):
    TypedBlockScheduler.policy = policy
    register_typed_planner(TypedBlockScheduler)
    capture_cls, captured = _capture_bodies()
    register_typed_planner(capture_cls)

    n = 128

    @cuda.jit
    def kernel(out, a, b):
        i = cuda.grid(1)
        if i < out.size:
            scratch = cuda.local.array(4, dtype=np.float32)
            x = a[i] * np.float32(2.0)
            y = b[i] + np.float32(3.0)
            scratch[0] = x * y
            scratch[1] = a[i] - b[i]
            scratch[2] = x + y
            scratch[3] = x * y - (a[i] - b[i])
            acc = np.float32(0.0)
            for j in range(4):
                acc += scratch[j] * np.float32(j + 1)
            out[i] = acc

    a = np.arange(n, dtype=np.float32)
    b = np.full(n, 0.5, dtype=np.float32)
    out = np.zeros(n, dtype=np.float32)
    kernel[2, 64](out, a, b)

    x = a * np.float32(2.0)
    y = b + np.float32(3.0)
    z = x * y
    w = a - b
    expected = z + w * 2 + (x + y) * 3 + (z - w) * 4
    np.testing.assert_allclose(out, expected, rtol=1e-6)
    return captured


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize(
    "policy", ["dfs", "anchor_dfs", "liveness", "longlived_dfs"]
)
def test_scheduled_kernels_stay_correct(isolated_typed_planners, policy):
    captured = _run_scheduled_kernel(policy)
    stats = captured["stats"]
    assert stats["policy"] == policy
    assert stats["blocks"] > 0
    # The synthetic kernel gives every policy freedom to move something.
    assert stats["reordered_blocks"] > 0
    assert stats["moved_statements"] > 0


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_source_policy_never_reorders(isolated_typed_planners):
    captured = _run_scheduled_kernel("source")
    stats = captured["stats"]
    assert stats["policy"] == "source"
    assert stats["reordered_blocks"] == 0
    assert stats["moved_statements"] == 0


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
def test_scheduling_preserves_statement_multiset(isolated_typed_planners):
    # One compile: pre-schedule capture registers first, post after.
    baseline_cls, baseline = _capture_bodies()
    register_typed_planner(baseline_cls)
    scheduled = _run_scheduled_kernel("dfs")

    scheduled_labels = {
        label for label in scheduled if isinstance(label, int)
    }
    baseline_labels = {
        label for label in baseline if isinstance(label, int)
    }
    assert scheduled_labels == baseline_labels
    moved = 0
    for label in scheduled_labels:
        scheduled_ids = [id(statement) for statement in scheduled[label]]
        baseline_ids = [id(statement) for statement in baseline[label]]
        assert sorted(scheduled_ids) == sorted(baseline_ids)
        moved += sum(
            1
            for before, after in zip(baseline_ids, scheduled_ids)
            if before != after
        )
    assert moved > 0


def _statement_accesses(statement):
    """Return (defs, uses) name sets mirroring the scheduler's view."""

    defs = set()
    uses = set()
    if isinstance(statement, ir.Assign):
        defs.add(statement.target.name)
        value = statement.value
        if isinstance(value, ir.Expr):
            uses |= {var.name for var in value.list_vars()}
        elif isinstance(value, ir.Var):
            uses.add(value.name)
    elif isinstance(statement, ir.Del):
        uses.add(statement.value)
    else:
        uses |= {var.name for var in statement.list_vars()}
    return defs, uses


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.parametrize(
    "policy", ["dfs", "anchor_dfs", "liveness", "longlived_dfs"]
)
def test_schedule_respects_dependencies(isolated_typed_planners, policy):
    """Every same-name def/use conflict keeps its original order."""

    baseline_cls, baseline = _capture_bodies()
    register_typed_planner(baseline_cls)
    scheduled = _run_scheduled_kernel(policy)
    for label, body in scheduled.items():
        if not isinstance(label, int):
            continue
        original = baseline[label]
        original_position = {
            id(statement): position
            for position, statement in enumerate(original)
        }
        accesses = {}
        for position, statement in enumerate(body):
            defs, uses = _statement_accesses(statement)
            for name in defs | uses:
                accesses.setdefault(name, []).append(
                    (position, statement, name in defs)
                )
        for name, members in accesses.items():
            for first_index in range(len(members)):
                for second_index in range(first_index + 1, len(members)):
                    one_pos, one_stmt, one_def = members[first_index]
                    two_pos, two_stmt, two_def = members[second_index]
                    if not (one_def or two_def):
                        continue  # use-use pairs commute
                    original_one = original_position[id(one_stmt)]
                    original_two = original_position[id(two_stmt)]
                    assert (original_one < original_two) == (
                        one_pos < two_pos
                    ), (
                        f"conflict on {name} in block {label} reordered: "
                        f"orig {original_one}/{original_two} -> "
                        f"new {one_pos}/{two_pos}"
                    )
