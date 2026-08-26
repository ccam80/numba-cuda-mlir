// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// RUN: llvm70-translate %s --dump-llvm 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-IR %s
// RUN: llvm70-translate %s --dump-ptx 2>&1 >/dev/null | FileCheck --check-prefix=CHECK-PTX %s

// Latch loop_annotation -> `!llvm.loop` ID node; same annotation, same node.

module {
  gpu.module @kernels [#nvvm_llvm70.target<chip = "sm_75">] {

    // CHECK-IR-LABEL: define ptx_kernel void @full_unroll
    // CHECK-IR: br i1 %{{.*}}, label %{{.*}}, label %{{.*}}, !llvm.loop [[FULL:![0-9]+]]
    llvm.func @full_unroll(%out: !llvm.ptr<1>) attributes {gpu.kernel} {
      %c0 = llvm.mlir.constant(0 : i64) : i64
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %c8 = llvm.mlir.constant(8 : i64) : i64
      %one = llvm.mlir.constant(1.0 : f32) : f32
      llvm.br ^header(%c0 : i64)
    ^header(%i: i64):
      %ptr = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %one, %ptr : f32, !llvm.ptr<1>
      %next = llvm.add %i, %c1 : i64
      %done = llvm.icmp "eq" %next, %c8 : i64
      llvm.cond_br %done, ^exit, ^header(%next : i64) {loop_annotation = #llvm.loop_annotation<unroll = <full = true>>}
    ^exit:
      llvm.return
    }

    // CHECK-IR-LABEL: define ptx_kernel void @unroll_count
    // CHECK-IR: br label %{{.*}}, !llvm.loop [[COUNT:![0-9]+]]
    llvm.func @unroll_count(%out: !llvm.ptr<1>, %n: i64) attributes {gpu.kernel} {
      %c0 = llvm.mlir.constant(0 : i64) : i64
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %one = llvm.mlir.constant(1.0 : f32) : f32
      llvm.br ^header(%c0 : i64)
    ^header(%i: i64):
      %done = llvm.icmp "sge" %i, %n : i64
      llvm.cond_br %done, ^exit, ^body
    ^body:
      %ptr = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %one, %ptr : f32, !llvm.ptr<1>
      %next = llvm.add %i, %c1 : i64
      llvm.br ^header(%next : i64) {loop_annotation = #llvm.loop_annotation<unroll = <count = 4 : i32, runtimeDisable = true>>}
    ^exit:
      llvm.return
    }

    // CHECK-IR-LABEL: define ptx_kernel void @two_latches
    // CHECK-IR: br label %{{.*}}, !llvm.loop [[DISABLE:![0-9]+]]
    // CHECK-IR: br label %{{.*}}, !llvm.loop [[DISABLE]]
    llvm.func @two_latches(%out: !llvm.ptr<1>, %n: i64) attributes {gpu.kernel} {
      %c0 = llvm.mlir.constant(0 : i64) : i64
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %c2 = llvm.mlir.constant(2 : i64) : i64
      %one = llvm.mlir.constant(1.0 : f32) : f32
      %two = llvm.mlir.constant(2.0 : f32) : f32
      llvm.br ^header(%c0 : i64)
    ^header(%i: i64):
      %done = llvm.icmp "sge" %i, %n : i64
      llvm.cond_br %done, ^exit, ^body
    ^body:
      %ptr = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      %next = llvm.add %i, %c1 : i64
      %rem = llvm.srem %i, %c2 : i64
      %odd = llvm.icmp "ne" %rem, %c0 : i64
      llvm.cond_br %odd, ^odd, ^even
    ^odd:
      llvm.store %one, %ptr : f32, !llvm.ptr<1>
      llvm.br ^header(%next : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
    ^even:
      llvm.store %two, %ptr : f32, !llvm.ptr<1>
      llvm.br ^header(%next : i64) {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
    ^exit:
      llvm.return
    }

    // CHECK-IR-LABEL: define ptx_kernel void @no_annotation
    // CHECK-IR-NOT: !llvm.loop
    // CHECK-IR: ret void
    llvm.func @no_annotation(%out: !llvm.ptr<1>, %n: i64) attributes {gpu.kernel} {
      %c0 = llvm.mlir.constant(0 : i64) : i64
      %c1 = llvm.mlir.constant(1 : i64) : i64
      %one = llvm.mlir.constant(1.0 : f32) : f32
      llvm.br ^header(%c0 : i64)
    ^header(%i: i64):
      %done = llvm.icmp "sge" %i, %n : i64
      llvm.cond_br %done, ^exit, ^body
    ^body:
      %ptr = llvm.getelementptr %out[%i] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, f32
      llvm.store %one, %ptr : f32, !llvm.ptr<1>
      %next = llvm.add %i, %c1 : i64
      llvm.br ^header(%next : i64)
    ^exit:
      llvm.return
    }
  }
}

// CHECK-IR-DAG: [[FULL]] = distinct !{[[FULL]], [[FULLPROP:![0-9]+]]}
// CHECK-IR-DAG: [[FULLPROP]] = !{!"llvm.loop.unroll.full"}
// CHECK-IR-DAG: [[COUNT]] = distinct !{[[COUNT]], [[COUNTPROP:![0-9]+]], [[RTPROP:![0-9]+]]}
// CHECK-IR-DAG: [[COUNTPROP]] = !{!"llvm.loop.unroll.count", i32 4}
// CHECK-IR-DAG: [[RTPROP]] = !{!"llvm.loop.unroll.runtime.disable"}
// CHECK-IR-DAG: [[DISABLE]] = distinct !{[[DISABLE]], [[DISABLEPROP:![0-9]+]]}
// CHECK-IR-DAG: [[DISABLEPROP]] = !{!"llvm.loop.unroll.disable"}

// The full-unroll hint is honoured by libnvvm: eight stores, no loop branch.
// CHECK-PTX-LABEL: .entry full_unroll
// CHECK-PTX-NOT: bra
// CHECK-PTX-COUNT-8: st.global.u32
// CHECK-PTX-NOT: bra
// CHECK-PTX: ret;
