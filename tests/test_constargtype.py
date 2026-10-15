# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir.ast_transforms import ConstevalError
from numba_cuda_mlir.cuda.experimental import constargtype, consteval
from numba_cuda_mlir.numba_cuda import types


@cuda.jit(device=True, inline=True)
def store(out, v):
    if constargtype(out).ndim == 1:
        out[0] = v
    else:
        out[0, 0] = v


def test_branch_on_argument_type_is_removed_before_typing():
    @cuda.jit
    def kernel(out, v):
        store(out, v)

    flat = cuda.to_device(np.zeros(2, dtype=np.int32))
    square = cuda.to_device(np.zeros((2, 2), dtype=np.int32))
    kernel[1, 1](flat, 5)
    kernel[1, 1](square, 7)
    np.testing.assert_array_equal(flat.copy_to_host(), [5, 0])
    np.testing.assert_array_equal(square.copy_to_host(), [[7, 0], [0, 0]])


def test_comparison_with_a_numba_type_selects_branch():
    @cuda.jit(device=True, inline=True)
    def kind(v):
        if constargtype(v) == types.float32:
            return 1
        return 2

    @cuda.jit
    def kernel(out, single, double):
        out[0] = kind(single)
        out[1] = kind(double)

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    kernel[1, 1](out, np.float32(1), np.float64(1))
    np.testing.assert_array_equal(out.copy_to_host(), [1, 2])


def test_nested_inlined_callee_sees_computed_argument_type():
    @cuda.jit(device=True, inline=True)
    def inner(out, v):
        out[0] = constargtype(v).bitwidth

    @cuda.jit(device=True, inline=True)
    def outer(out, a):
        inner(out, a * 2)

    @cuda.jit
    def kernel(out, a):
        outer(out, a)
        out[1] = constargtype(a * 2).bitwidth

    out = cuda.to_device(np.zeros(2, dtype=np.int64))
    kernel[1, 1](out, np.float32(1.5))
    inlined, direct = out.copy_to_host()
    assert inlined == direct


def test_kernel_argument_type():
    @cuda.jit
    def kernel(out):
        out[0] = constargtype(out).ndim

    out = cuda.to_device(np.zeros((1, 1, 1), dtype=np.int64))
    kernel[1, 1](out)
    assert out.copy_to_host()[0, 0, 0] == out.ndim


def test_consteval_comparing_inlinee_parameter_raises():
    @cuda.jit(device=True, inline=True)
    def pick(out, x):
        if consteval(x == 0):
            out[0] = 111
        else:
            out[0] = 222

    @cuda.jit(experimental_ast_transforms=True)
    def kernel(out):
        pick(out, 0)

    out = cuda.to_device(np.zeros(1, dtype=np.int32))
    with pytest.raises(ConstevalError, match="constargtype"):
        kernel[1, 1](out)
