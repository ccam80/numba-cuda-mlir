# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Experimental Numba-CUDA-MLIR features.

When a kernel's defining module has imported from this package, ``@cuda.jit``
auto-enables ``experimental_ast_transforms=True`` for that kernel (unless the
user explicitly passes ``experimental_ast_transforms=False``).
"""

import importlib as _importlib

from numba_cuda_mlir.cuda import inline_ptx  # noqa: F401
from numba_cuda_mlir.cuda.experimental.struct import struct  # noqa: F401
from numba_cuda_mlir.cuda.experimental.union import union  # noqa: F401


def __getattr__(name):
    _lazy = {
        "intrin": "numba_cuda_mlir.cuda.intrin",
        "tcgen05_descriptors": "numba_cuda_mlir.cuda.experimental.tcgen05_descriptors",
    }
    if name in _lazy:
        return _importlib.import_module(_lazy[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _ConstevalContextManager:
    """Context manager for consteval blocks -- transformed away by AST passes."""

    def __enter__(self):
        raise RuntimeError(
            "consteval() block was not transformed at compile time.\n"
            "Ensure experimental_ast_transforms is enabled:\n"
            "    from numba_cuda_mlir.cuda.experimental import consteval\n"
            "    @cuda.jit(experimental_ast_transforms=True)"
        )

    def __exit__(self, *args):
        pass


def consteval(value=None):
    """
    Evaluate an expression at compile time, or mark a block for compile-time
    execution.

    Usage as expression (returns the compile-time value)::

        x = consteval(GLOBAL_CONST * 2)

    Usage in a loop (unrolls each compile-time iteration)::

        for i in consteval(range(10)):
            array[i] = i

    Usage as context manager (executes block at compile time)::

        with consteval():
            config = load_config()
            N = config["block_size"]

    Inlined device functions (``inline=True``) are transformed when the calling
    kernel enables the AST transforms; inside them ``current_target_options()``
    is the calling kernel's options and a consteval that names a parameter
    raises. Use ``constargtype`` for a parameter's type.
    """
    if value is None:
        return _ConstevalContextManager()
    raise RuntimeError(
        "consteval() was not transformed at compile time.\n"
        "Ensure experimental_ast_transforms is enabled:\n"
        "    from numba_cuda_mlir.cuda.experimental import consteval\n"
        "    @cuda.jit(experimental_ast_transforms=True)"
    )


def constargtype(value):
    """
    Return the Numba type of an argument at compile time.

    The compiler replaces the call with the argument's type after inlining and
    folds the attribute reads, comparisons and calls that use it, so a branch
    on the result is removed before typing::

        @cuda.jit(device=True, inline=True)
        def store(out, v):
            if constargtype(out).ndim == 1:
                out[0] = v
            else:
                out[0, 0] = v

    It works in kernels and in inlined device functions alike.
    """
    raise RuntimeError("constargtype() was not replaced at compile time.")


class _CurrentTargetOptionsMarker:
    def __repr__(self):
        return "current_target_options()"


def current_target_options() -> dict:
    """
    Return the current kernel's target options as a dictionary.

    Only usable inside ``consteval()`` expressions::

        @cuda.jit(chip="sm_90")
        def kernel(arr):
            chip = consteval(current_target_options()["chip"])
    """
    return _CurrentTargetOptionsMarker()


def local_array_from(iterable, dtype):
    """
    Create a local array from a generator expression or iterable.

    Transformed at AST level into ``cuda.local_array`` + assignment loop::

        arr = local_array_from((i + 1 for i in indices), dtype=np.float32)
    """
    pass


__all__ = [
    "consteval",
    "constargtype",
    "current_target_options",
    "inline_ptx",
    "intrin",
    "local_array_from",
    "struct",
    "tcgen05_descriptors",
    "union",
]
