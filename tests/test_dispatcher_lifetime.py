# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dispatchers and their native launch objects are freed once dropped."""

import gc
import weakref

import numpy as np

from numba_cuda_mlir import _cext, cuda
from numba_cuda_mlir import descriptor as descriptor_mod

_Py_TPFLAGS_HAVE_GC = 1 << 14


def _make_kernel():
    @cuda.jit(device=True)
    def helper(x):
        return x + 1

    @cuda.jit
    def kern(a):
        a[0] = helper(a[0])

    return kern, helper


def test_native_launch_types_participate_in_gc():
    assert _cext.KernelDispatcher.__flags__ & _Py_TPFLAGS_HAVE_GC
    assert _cext.LaunchConfiguration.__flags__ & _Py_TPFLAGS_HAVE_GC


def test_dropped_kernel_is_collected_after_launch():
    kern, helper = _make_kernel()
    a = cuda.to_device(np.zeros(1, dtype=np.int64))
    kern[1, 1](a)
    cuda.synchronize()
    assert a.copy_to_host()[0] == 1

    kern_ref = weakref.ref(kern)
    del kern, helper
    gc.collect()

    assert kern_ref() is None


def test_dropped_kernel_dispatcher_teardown_after_launch(monkeypatch):
    # Zero retention frees the launched native dispatcher on recompile().
    monkeypatch.setattr(descriptor_mod, "_OLD_DISPATCHER_RETAIN_LIMIT", 0)
    kern, _ = _make_kernel()
    a = cuda.to_device(np.zeros(1, dtype=np.int64))
    kern[1, 1](a)
    kern.recompile()
    gc.collect()
    kern[1, 1](a)
    cuda.synchronize()
    assert a.copy_to_host()[0] == 2
