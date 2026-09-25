..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _typed-planners:

Typed whole-function planners
=============================

Typed planners run on the fully inlined, legalized, typed Numba IR
immediately before lowering to MLIR; ``state.typemap`` and
``state.calltypes`` cover the whole function body.  This is the hook
an embedding library uses to apply its own whole-kernel IR passes —
for example a statement scheduler that reorders each basic block
under a dependency DAG — without carrying compiler passes in this
package.

.. py:function:: register_typed_planner(planner_cls)

   Register a :py:class:`TypedWholeFunctionPlanner` subclass before
   compiling any dispatcher that needs it.

.. py:class:: TypedWholeFunctionPlanner

   Base class for typed planners.  Subclasses implement ``run()`` and
   return a Boolean indicating whether they changed the IR.  The
   registry rebuilds ``func_ir`` definitions and re-verifies every
   block after a modifying planner.

   .. py:attribute:: cache_safe

      Whether persistent dispatch-cache loads and saves may proceed
      while this planner is registered (default ``False``).  Set it
      to ``True`` only when the planner's effect is deterministic and
      the embedder keys its own cache identity on the planner's
      configuration; while any registered planner is not cache-safe,
      dispatch caching is disabled.
