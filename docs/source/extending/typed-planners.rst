..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _typed-planners:

Typed whole-function planners
=============================

Typed planners run on the fully inlined, legalized, typed Numba IR
immediately before lowering to MLIR; ``state.typemap`` and
``state.calltypes`` cover the whole function body.

.. py:function:: register_typed_planner(planner_cls)

   Register a :py:class:`TypedWholeFunctionPlanner` subclass before
   compiling any dispatcher that needs it.  While any planner registry
   is populated, persistent dispatch-cache loads and saves are
   disabled.

.. py:class:: TypedWholeFunctionPlanner

   Base class for typed planners.  Subclasses implement ``run()`` and
   return a Boolean indicating whether they changed the IR.  The
   registry rebuilds ``func_ir`` definitions and re-verifies every
   block after a modifying planner.

Intra-block statement scheduling
--------------------------------

:py:class:`numba_cuda_mlir.typed_block_scheduler.TypedBlockScheduler`
is a typed planner that reorders the statements of each basic block
without changing the CFG, the SSA names, or the statement set.
Legality is a per-block dependency DAG: SSA flow edges, per-element
memory chains keyed on ``(alias root, constant index)``, effect
barriers, and ``Del`` lifetime pins.  The ``policy`` class attribute
(default from ``NUMBA_CUDA_MLIR_BLOCK_SCHEDULE``) selects the emitted
order:

``source``
   Keep the original order (measurement control).

``dfs``
   Roots-first predecessor postorder: every externally consumed value
   is emitted together with the chain that computes it.

``liveness``
   Greedy list schedule preferring statements that close the most live
   values.

``longlived_dfs``
   The ``dfs`` emission with chains feeding block-live-out values
   emitted first.

``inject``
   Apply explicit per-block orders from the JSON file named by
   ``NUMBA_CUDA_MLIR_BLOCK_SCHEDULE_ORDER`` (validated against the
   dependency DAG).

Setting ``NUMBA_CUDA_MLIR_BLOCK_SCHEDULE_DUMP`` writes the dependency
graph of every large block to a gzip JSON file for offline ordering
experiments; the chosen order can be injected back with the ``inject``
policy.  Compile metadata records per-compile statistics under the
``typed_block_scheduler`` key, including the modeled live-scalar peak
of the source and scheduled orders.
