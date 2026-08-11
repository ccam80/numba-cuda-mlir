# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the cached callee IR used by inline="always" inlining."""

import gc
import weakref

import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.compiler import CUDACompiler
from numba_cuda_mlir.numba_cuda.core.inline_closurecall import InlineWorker
from numba_cuda_mlir.numba_cuda.descriptor import cuda_target
from numba_cuda_mlir.numba_cuda.flags import Flags

GLOBAL_SCALE = 2.0
WORKER_OFFSET = 5.0


def _make_worker():
    flags = Flags()
    tyctx = cuda_target.typing_context
    tgctx = cuda_target.target_context
    compiler_inst = CUDACompiler(tyctx, tgctx, None, None, None, flags, None)
    return InlineWorker(tyctx, tgctx, {}, compiler_inst, flags, None)


def test_global_change_recompiles_inlined_callee():
    global GLOBAL_SCALE
    GLOBAL_SCALE = 2.0
    try:

        @cuda.jit(device=True, inline="always")
        def callee(x):
            return x * GLOBAL_SCALE

        @cuda.jit
        def kernel(a):
            a[0] = callee(a[0])

        a = np.ones(1, dtype=np.float64)
        kernel[1, 1](a)
        assert a[0] == 2.0

        GLOBAL_SCALE = 10.0
        b = np.ones(1, dtype=np.float32)
        kernel[1, 1](b)
        assert b[0] == 10.0
    finally:
        GLOBAL_SCALE = 2.0


def test_closure_cell_change_recompiles_inlined_callee():
    def make_callee():
        c = 3.0

        @cuda.jit(device=True, inline="always")
        def callee(x):
            return x + c

        def set_c(value):
            nonlocal c
            c = value

        return callee, set_c

    callee, set_c = make_callee()

    @cuda.jit
    def kernel(a):
        a[0] = callee(a[0])

    a = np.zeros(1, dtype=np.float64)
    kernel[1, 1](a)
    assert a[0] == 3.0

    set_c(7.0)
    b = np.zeros(1, dtype=np.float32)
    kernel[1, 1](b)
    assert b[0] == 7.0


def test_cache_hit_skips_untyped_pipeline():
    global WORKER_OFFSET
    WORKER_OFFSET = 5.0
    try:

        def callee(x):
            return x + WORKER_OFFSET

        worker = _make_worker()
        runs = []
        run_untyped_passes = InlineWorker.run_untyped_passes

        def counting_run(self, func, enable_ssa=False):
            runs.append(func)
            return run_untyped_passes(self, func, enable_ssa)

        InlineWorker.run_untyped_passes = counting_run
        try:
            worker._fresh_callee_ir(callee)
            worker._fresh_callee_ir(callee)
            assert len(runs) == 1

            WORKER_OFFSET = 6.0
            worker._fresh_callee_ir(callee)
            assert len(runs) == 2

            worker._fresh_callee_ir(callee)
            assert len(runs) == 2
        finally:
            InlineWorker.run_untyped_passes = run_untyped_passes
    finally:
        WORKER_OFFSET = 5.0


def test_cache_released_with_function():
    def make_callee():
        def callee(x):
            return x + 1.0

        return callee

    callee = make_callee()
    worker = _make_worker()
    worker._fresh_callee_ir(callee)

    ref = weakref.ref(callee)
    del callee
    gc.collect()
    assert ref() is None
