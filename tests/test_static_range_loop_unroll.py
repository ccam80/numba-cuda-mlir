# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``range`` loops with compile-time trip counts carry full-unroll metadata."""

import re

import numpy as np
import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.loop_annotations import STATIC_RANGE_LOOP_LOC_PREFIX
from numba_cuda_mlir.lowering_utilities import context, try_fold_int_constant
from numba_cuda_mlir.types import float32, int32

N = 10
STAGES = 7
ACC = (STAGES - 1) * N

_ANNOTATED_LATCH = re.compile(r"llvm\.(?:cond_)?br\b.*\{loop_annotation = ")


def _annotated_latches(kernel):
    (mlir_text,) = kernel.inspect_mlir_optimized().values()
    assert STATIC_RANGE_LOOP_LOC_PREFIX not in mlir_text, "header tag leaked past annotation"
    return len(_ANNOTATED_LATCH.findall(mlir_text)), mlir_text


def test_constant_trip_range_loop_is_annotated():
    @cuda.jit
    def kernel(out):
        for i in range(8):
            out[i] = i

    out = cuda.to_device(np.zeros(8, dtype=np.int32))
    kernel[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), np.arange(8))

    count, mlir_text = _annotated_latches(kernel)
    assert count == 1
    assert "#llvm.loop_unroll<full = true>" in mlir_text


def test_runtime_trip_range_loop_is_not_annotated():
    @cuda.jit
    def kernel(out, n):
        for i in range(n):
            out[i] = i

    out = cuda.to_device(np.zeros(8, dtype=np.int32))
    kernel[1, 1](out, 8)
    np.testing.assert_array_equal(out.copy_to_host(), np.arange(8))

    count, mlir_text = _annotated_latches(kernel)
    assert count == 0
    assert "loop_annotation" not in mlir_text


def test_constant_trip_from_global_expressions_is_annotated():
    """Bounds built from frozen globals (``STAGES - 1``, ``N // 2 + 1``) fold."""

    @cuda.jit
    def kernel(out):
        acc = 0
        for prev in range(STAGES - 1):
            acc += prev
        for i in range(1, N // 2 + 1, 2):
            acc += i
        out[0] = acc

    out = cuda.to_device(np.zeros(1, dtype=np.int32))
    kernel[1, 1](out)
    assert out.copy_to_host()[0] == sum(range(STAGES - 1)) + sum(range(1, N // 2 + 1, 2))

    count, _ = _annotated_latches(kernel)
    assert count == 2


def test_nested_constant_loops_are_each_annotated():
    @cuda.jit
    def kernel(out):
        acc = float32(0.0)
        for a in range(3):
            for b in range(4):
                for c in range(5):
                    acc += float32(a * 20 + b * 5 + c)
        out[0] = acc

    out = cuda.to_device(np.zeros(1, dtype=np.float32))
    kernel[1, 1](out)
    expected = sum(a * 20 + b * 5 + c for a in range(3) for b in range(4) for c in range(5))
    assert out.copy_to_host()[0] == expected

    count, _ = _annotated_latches(kernel)
    assert count == 3


def test_constant_loop_with_break_and_continue():
    """Loops with extra exits and latches are still annotated and correct."""

    @cuda.jit
    def kernel(out):
        acc = 0
        for i in range(16):
            if i % 3 == 0:
                continue
            if i > 10:
                break
            acc += i
        out[0] = acc

    out = cuda.to_device(np.zeros(1, dtype=np.int32))
    kernel[1, 1](out)
    assert out.copy_to_host()[0] == sum(i for i in range(11) if i % 3 != 0)

    count, _ = _annotated_latches(kernel)
    assert count >= 1


def test_empty_and_negative_step_ranges_still_run():
    @cuda.jit
    def kernel(out):
        acc = 0
        for i in range(5, 5):
            acc += 100
        for i in range(10, 3, -2):
            acc += i
        out[0] = acc

    out = cuda.to_device(np.zeros(1, dtype=np.int32))
    kernel[1, 1](out)
    assert out.copy_to_host()[0] == sum(range(10, 3, -2))


def _make_issue8_kernel(rhs_in_shared, **jit_kwargs):
    """ERK-style accumulate nest reading a local array or a dynamic-shared slice."""
    coeffs = tuple(
        tuple(np.float32(0.1 * (i + 1) + 0.01 * j) for j in range(STAGES)) for i in range(STAGES)
    )

    @cuda.jit(device=True, inline=True, **jit_kwargs)
    def rhs_fn(y, out):
        for i in range(N):
            out[i] = y[(i + 1) % N] * (y[(i + 2) % N] - y[(i + 9) % N]) - y[i]

    @cuda.jit(device=True, inline=True, **jit_kwargs)
    def step(y, out, shared, dt):
        if rhs_in_shared:
            rhs = shared[0:N]
        else:
            rhs = cuda.local.array(N, float32)
        acc = cuda.local.array(ACC, float32)
        rhs_fn(y, rhs)
        for i in range(ACC):
            acc[i] = float32(0.0)
        for prev in range(STAGES - 1):
            col = coeffs[prev]
            for succ in range(STAGES - 1):
                c = col[succ + 1]
                row = succ * N
                for i in range(N):
                    acc[row + i] += c * rhs[i]
            base = prev * N
            for i in range(N):
                acc[base + i] = acc[base + i] * dt + y[i]
            rhs_fn(acc[base : base + N], rhs)
        for i in range(N):
            out[i] = y[i] + dt * rhs[i]

    @cuda.jit(**jit_kwargs)
    def kernel(ys, outs, steps):
        i = cuda.grid(1)
        if i >= ys.shape[0]:
            return
        shared_all = cuda.shared.array(0, dtype=float32)
        lo = int32(cuda.threadIdx.x * N)
        shared = shared_all[lo : lo + N]
        y = cuda.local.array(N, float32)
        out = cuda.local.array(N, float32)
        for k in range(N):
            y[k] = ys[i, k]
        for _ in range(steps):
            step(y, out, shared, float32(1e-3))
            for k in range(N):
                y[k] = out[k]
        for k in range(N):
            outs[i, k] = y[k]

    return kernel


def _run_issue8(kernel, steps):
    ys = (8.0 + 0.01 * np.arange(64 * N, dtype=np.float32)).reshape(64, N)
    d_ys = cuda.to_device(ys)
    d_out = cuda.device_array_like(ys)
    kernel[2, 32, 0, 32 * N * 4](d_ys, d_out, steps)
    cuda.synchronize()
    return d_out.copy_to_host()


def test_shared_operand_unrolls_like_local_operand():
    """Shared and local operands give the same PTX shape: no local depot, one loop."""
    local = _make_issue8_kernel(False, fastmath=True)
    shared = _make_issue8_kernel(True, fastmath=True)
    _run_issue8(local, 1)
    _run_issue8(shared, 1)

    (ptx_local,) = local.inspect_asm().values()
    (ptx_shared,) = shared.inspect_asm().values()

    assert "__local_depot" not in ptx_local
    assert "__local_depot" not in ptx_shared

    fma = re.compile(r"\bfma\.rn")
    bra = re.compile(r"\bbra\b")
    assert len(fma.findall(ptx_shared)) == len(fma.findall(ptx_local))
    assert len(bra.findall(ptx_shared)) == len(bra.findall(ptx_local))
    # Only the runtime `steps` loop and the grid-guard remain as branches.
    assert len(bra.findall(ptx_shared)) <= 4

    count_local, _ = _annotated_latches(local)
    count_shared, _ = _annotated_latches(shared)
    assert count_local == count_shared == 11


def test_issue8_results_match_between_operand_memory_spaces():
    local = _make_issue8_kernel(False)
    shared = _make_issue8_kernel(True)
    np.testing.assert_allclose(_run_issue8(shared, 50), _run_issue8(local, 50), rtol=1e-5)


@pytest.mark.parametrize(
    "build, expected",
    [
        (lambda c: c(7), 7),
        (lambda c: arith.subi(c(7), c(1)), 6),
        (lambda c: arith.addi(arith.muli(c(6), c(10)), c(1)), 61),
        (lambda c: arith.divsi(c(-7), c(2)), -3),
        (lambda c: arith.remsi(c(-7), c(2)), -1),
        (lambda c: arith.floordivsi(c(-7), c(2)), -4),
        (lambda c: arith.ceildivsi(c(7), c(2)), 4),
        (lambda c: arith.maxsi(c(3), c(9)), 9),
        (lambda c: arith.minsi(c(3), c(9)), 3),
        (lambda c: arith.trunci(T.i8(), c(300)), 44),
        (lambda c: arith.extui(T.i64(), arith.trunci(T.i8(), c(-1))), 255),
        (lambda c: arith.extsi(T.i64(), arith.trunci(T.i8(), c(-1))), -1),
        (lambda c: arith.index_cast(T.index(), c(12)), 12),
        (lambda c: arith.divsi(c(7), c(0)), None),
    ],
)
def test_try_fold_int_constant(build, expected):
    with context.get_context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):

            def c(value):
                return arith.constant(T.i64(), value)

            assert try_fold_int_constant(build(c)) == expected


def test_try_fold_int_constant_runtime_value_is_none():
    with context.get_context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fn = ir.Operation.create(
                "func.func",
                attributes={
                    "sym_name": ir.StringAttr.get("f"),
                    "function_type": ir.TypeAttr.get(ir.FunctionType.get([T.i64()], [])),
                },
                regions=1,
            )
            block = fn.regions[0].blocks.append(T.i64())
            with ir.InsertionPoint(block):
                arg = block.arguments[0]
                assert try_fold_int_constant(arg) is None
                assert try_fold_int_constant(arith.addi(arg, arith.constant(T.i64(), 1))) is None
                ir.Operation.create("func.return")
