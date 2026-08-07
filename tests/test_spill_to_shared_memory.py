# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the spill_to_shared_memory dispatcher option (CUDA 13)."""

import numpy as np
import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir.tools import get_cuda_runtime_version

try:
    _has_cuda13 = get_cuda_runtime_version() >= (13, 0)
except Exception:
    _has_cuda13 = False

requires_cuda13 = pytest.mark.skipif(not _has_cuda13, reason="requires CUDA 13 toolkit")


def _increment_kernel_source():
    @cuda.jit(spill_to_shared_memory=True, lto=True, launch_bounds=128)
    def kernel(arr):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] += 1

    return kernel


@requires_cuda13
def test_pragma_in_lto_ptx_and_kernel_runs():
    kernel = _increment_kernel_source()
    arr = cuda.to_device(np.arange(10, dtype=np.int32))
    kernel[1, 10](arr)
    np.testing.assert_array_equal(arr.copy_to_host(), np.arange(1, 11, dtype=np.int32))
    lto_ptx = next(iter(kernel.inspect_lto_ptx().values()))
    assert '.pragma "enable_smem_spilling";' in lto_ptx


@requires_cuda13
def test_pragma_absent_by_default():
    @cuda.jit(lto=True)
    def kernel(arr):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = i

    arr = cuda.device_array(10, dtype=np.int32)
    kernel[1, 10](arr)
    lto_ptx = next(iter(kernel.inspect_lto_ptx().values()))
    assert "enable_smem_spilling" not in lto_ptx


def test_requires_lto():
    with pytest.raises(ValueError, match="requires lto=True"):

        @cuda.jit(spill_to_shared_memory=True)
        def kernel(arr):
            arr[0] = 0


def test_conflicts_with_debug():
    with pytest.raises(ValueError, match="not supported with debug=True"):

        @cuda.jit(spill_to_shared_memory=True, lto=True, debug=True, opt=False)
        def kernel(arr):
            arr[0] = 0


def test_requires_cuda13(monkeypatch):
    monkeypatch.setattr("numba_cuda_mlir.tools.get_cuda_runtime_version", lambda: (12, 9))
    with pytest.raises(RuntimeError, match="requires CUDA 13.0 or later"):

        @cuda.jit("(int32[:],)", spill_to_shared_memory=True, lto=True)
        def kernel(arr):
            i = cuda.grid(1)
            if i < arr.size:
                arr[i] = i


def test_modern_path_mlir_injection():
    """The sm_100+ path injects the pragma as MLIR inline asm at kernel entry."""
    from numba_cuda_mlir._mlir import ir
    from numba_cuda_mlir._mlir.dialects import gpu
    from numba_cuda_mlir.lowering_utilities import context
    from numba_cuda_mlir.mlir_optimization import _inject_smem_spill_pragma

    source = """
    module {
      gpu.module @kernels {
        llvm.func @k(%arg0: !llvm.ptr<1>) attributes {gpu.kernel} {
          llvm.return
        }
        llvm.func @d(%arg0: f32) -> f32 {
          llvm.return %arg0 : f32
        }
      }
    }
    """
    with context.get_context():
        module = ir.Module.parse(source)
        gpu_mod = next(op for op in module.body if isinstance(op, gpu.GPUModuleOp))
        _inject_smem_spill_pragma(gpu_mod)
        text = str(module)
    kernel_body, device_body = text.split("llvm.func @d")
    assert 'llvm.inline_asm has_side_effects ".pragma \\22enable_smem_spilling\\22;"' in kernel_body
    assert "enable_smem_spilling" not in device_body
