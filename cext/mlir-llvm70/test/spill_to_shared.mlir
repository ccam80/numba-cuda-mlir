// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --spill-to-shared --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --spill-to-shared --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-OFF %s

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_80">] {

    // Kernels get the pragma as their first instruction.
    llvm.func @spill_kernel(%arg0: !llvm.ptr<1>) attributes {gpu.kernel, nvvm.maxntid = array<i32: 128>} {
      %c = llvm.mlir.constant(1.0 : f32) : f32
      llvm.store %c, %arg0 : f32, !llvm.ptr<1>
      llvm.return
    }

    // Device functions do not get the pragma.
    llvm.func @device_fn(%arg0: f32) -> f32 {
      %r = llvm.fadd %arg0, %arg0 : f32
      llvm.return %r : f32
    }
  }
}

// CHECK-IR: define ptx_kernel void @spill_kernel
// CHECK-IR-NEXT: call void asm sideeffect ".pragma \22enable_smem_spilling\22;", ""()
// CHECK-IR: define float @device_fn
// CHECK-IR-NOT: enable_smem_spilling

// CHECK-PTX: .visible .entry spill_kernel
// CHECK-PTX: .pragma "enable_smem_spilling";

// CHECK-OFF-NOT: enable_smem_spilling
