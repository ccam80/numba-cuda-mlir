# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from numba_cuda_mlir import cuda, mlir_compiler, types
from numba_cuda_mlir.ast_transforms import ConstevalError
from numba_cuda_mlir.cuda.experimental import consteval, current_target_options
from numba_cuda_mlir.errors import TypingError
from numba_cuda_mlir.numba_cuda.misc.special import literally


def _kernel_chip(kernel) -> str:
    (cres,) = kernel.overloads.values()
    return cres.metadata["targetoptions"]["chip"]


def test_inlined_callee_consteval_loop():
    n = 4

    @cuda.jit(device=True, inline=True)
    def fill(out, base):
        for i in consteval(range(n)):
            out[i] = base + i

    @cuda.jit
    def kernel(out):
        fill(out, 10)

    out = cuda.to_device(np.zeros(8, dtype=np.int32))
    kernel[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [10, 11, 12, 13, 0, 0, 0, 0])


@pytest.mark.xfail(strict=True, reason="fold_arguments cannot bind keyword-only arguments")
def test_inlined_callee_keyword_only_default():
    @cuda.jit(device=True, inline=True)
    def fill(out, *, base=7):
        for i in consteval(range(2)):
            out[i] = base + i

    @cuda.jit
    def kernel(out):
        fill(out)

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    kernel[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [7, 8])


def test_nested_inlined_callees_transform():
    n = 3

    @cuda.jit(device=True, inline=True)
    def inner(out, base):
        for i in consteval(range(n)):
            out[i] = base * consteval(10**i)

    @cuda.jit(device=True, inline=True)
    def outer(out):
        inner(out, 2)
        out[n] = consteval(n * 100)

    @cuda.jit
    def kernel(out):
        outer(out)

    out = cuda.to_device(np.zeros(4, dtype=np.int32))
    kernel[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [2, 20, 200, 300])


def test_inlined_callee_sees_caller_target_options():
    @cuda.jit(device=True, inline=True)
    def inner(out):
        chip = consteval(current_target_options()["chip"])
        out[1] = consteval(int(chip[3:]))

    @cuda.jit(device=True, inline=True)
    def outer(out):
        chip = consteval(current_target_options()["chip"])
        out[0] = consteval(int(chip[3:]))
        inner(out)

    @cuda.jit
    def kernel(out):
        outer(out)

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    kernel[1, 1](out)
    expected = int(_kernel_chip(kernel)[3:])
    np.testing.assert_array_equal(out.copy_to_host(), [expected, expected])


def test_inlined_callee_parameter_resolves_to_type():
    @cuda.jit(device=True, inline=True)
    def callee(out):
        out[0] = consteval(out.ndim)

    @cuda.jit
    def kernel(out):
        callee(out)

    out = cuda.to_device(np.zeros(1, dtype=np.int32))
    kernel[1, 1](out)
    assert out.copy_to_host()[0] == out.ndim


def _inlined_and_called(body):
    """The same device function compiled to be inlined and to be called."""
    return cuda.jit(device=True, inline=True)(body), cuda.jit(device=True)(body)


def _assert_inlined_matches_called(kernel, *args):
    out = cuda.to_device(np.zeros(2, dtype=np.int64))
    kernel[1, 1](out, *args)
    inlined, called = out.copy_to_host()
    assert inlined == called


def test_inlined_callee_computed_argument_types_match_call():
    def body(out, i, v):
        out[i] = consteval(v.bitwidth)

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out, a):
        inlined(out, 0, a * 2)
        called(out, 1, a * 2)

    _assert_inlined_matches_called(kernel, np.float32(1.5))


def test_inlined_callee_argument_from_inlined_call_matches_call():
    @cuda.jit(device=True, inline=True)
    def widen(a):
        return a * 2.0

    def body(out, i, v):
        out[i] = consteval(v.bitwidth)

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out, a):
        v = widen(a)
        inlined(out, 0, v)
        called(out, 1, v)

    _assert_inlined_matches_called(kernel, np.float32(1.5))


def test_inlined_callee_omitted_default_matches_call():
    def body(out, i, n=4):
        out[i] = consteval(isinstance(n, types.Omitted))

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out):
        inlined(out, 0)
        called(out, 1)

    _assert_inlined_matches_called(kernel)


def test_nested_inlined_callee_parameter_types():
    @cuda.jit(device=True, inline=True)
    def inner(out, v):
        out[0] = consteval(v.bitwidth)

    @cuda.jit(device=True, inline=True)
    def outer(out, a):
        inner(out, a * 2)

    @cuda.jit
    def kernel(out, a):
        outer(out, a)

    out = cuda.to_device(np.zeros(1, dtype=np.int64))
    kernel[1, 1](out, np.float32(1.5))
    assert out.copy_to_host()[0] == types.float64.bitwidth


def test_inlined_callee_literally_matches_call():
    def body(out, i, n):
        literally(n)
        out[i] = consteval(isinstance(n, types.IntegerLiteral))

    inlined, called = _inlined_and_called(body)

    @cuda.jit
    def kernel(out, n):
        inlined(out, 0, n)
        called(out, 1, n)

    _assert_inlined_matches_called(kernel, 3)


def test_recursive_callee_compiles_under_transforms():
    @cuda.jit(device=True, inline=True)
    def triangle(n):
        if n <= 0:
            return 0
        return n + triangle(n - 1)

    @cuda.jit
    def kernel(out, n):
        out[0] = triangle(n)

    out = cuda.to_device(np.zeros(1, dtype=np.int64))
    kernel[1, 1](out, 4)
    assert out.copy_to_host()[0] == 4 + 3 + 2 + 1


_shadowed = 3


def test_inlined_callee_parameter_shadows_global():
    @cuda.jit(device=True, inline=True)
    def callee(out, _shadowed):
        for i in consteval(range(_shadowed)):
            out[i] = 1

    @cuda.jit
    def kernel(out):
        callee(out, 5)

    with pytest.raises(ConstevalError, match="Cannot evaluate consteval argument"):
        kernel.compile("void(int32[:])")


def test_caller_option_controls_transform():
    n = 2

    @cuda.jit(device=True, inline=True, experimental_ast_transforms=False)
    def callee_off(out):
        for i in consteval(range(n)):
            out[i] = 1

    @cuda.jit(experimental_ast_transforms=True)
    def kernel_on(out):
        callee_off(out)

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    kernel_on[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [1, 1])

    @cuda.jit(device=True, inline=True, experimental_ast_transforms=True)
    def callee_on(out):
        for i in consteval(range(n)):
            out[i] = 1

    @cuda.jit(experimental_ast_transforms=False)
    def kernel_off(out):
        callee_on(out)

    with pytest.raises(TypingError, match="consteval"):
        kernel_off.compile("void(int32[:])")


def test_rejected_cost_model_does_not_transform(monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("rejected inlinee was transformed")

    monkeypatch.setattr(mlir_compiler, "transform_inline_callee", fail_if_called)

    def never_inline(expr, caller_info, callee_info):
        return False

    @cuda.jit(device=True, inline=never_inline)
    def callee(out):
        out[0] = 1

    @cuda.jit
    def kernel(out):
        callee(out)

    kernel.compile("void(int32[:])")
