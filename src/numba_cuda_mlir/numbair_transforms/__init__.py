# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Numba IR transformation passes for numba_cuda_mlir
from numba_cuda_mlir.numba_cuda.core.compiler_lock import global_compiler_lock
from numba_cuda_mlir.numba_cuda.core.untyped_passes import (
    FunctionPass,
    register_pass,
    TransformLiteralUnrollConstListToTuple,
    IterLoopCanonicalization,
    RewriteSemanticConstants,
    MixedContainerUnroller,
    GenericRewrites,
    InlineInlinables,
)
from numba_cuda_mlir.numba_cuda.core import untyped_passes as untyped_passes_module
from numba_cuda_mlir.numba_cuda.core.typed_passes import PartialTypeInference
from numba_cuda_mlir.numba_cuda.core import ir
from numba_cuda_mlir.numba_cuda.misc.special import literal_unroll
from numba_cuda_mlir._whole_function_planners import _planner_registry
from collections import Counter

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import errors
from numba_cuda_mlir.numba_cuda.core.analysis import dead_branch_prune
from numba_cuda_mlir.numba_cuda.core.ir_utils import build_definitions, get_definition, guard


@register_pass(mutates_CFG=True, analysis_only=False)
class NumbaCudaMlirLiteralUnroll(FunctionPass):
    """
    Implement the literal_unroll semantics.
    This is a numba_cuda_mlir-specific version that accepts both numba.misc.special.literal_unroll
    and numba.cuda.misc.special.literal_unroll.
    """

    _name = "numba_cuda_mlir_literal_unroll"

    def __init__(self):
        FunctionPass.__init__(self)

    def _is_literal_unroll(self, value):
        """Check if value is either version of literal_unroll."""
        if value is literal_unroll:
            return True
        return False

    def run_pass(self, state):
        # Determine whether to even attempt this pass... if there's no
        # `literal_unroll` as a global or as a freevar then just skip.
        found = False
        func_ir = state.func_ir
        for blk in func_ir.blocks.values():
            for asgn in blk.find_insts(ir.Assign):
                if isinstance(asgn.value, (ir.Global, ir.FreeVar)):
                    if self._is_literal_unroll(asgn.value.value):
                        found = True
                        break
            if found:
                break
        if not found:
            return False

        # run as subpipeline
        from numba_cuda_mlir.numba_cuda.core.compiler_machinery import PassManager

        pm = PassManager("literal_unroll_subpipeline")
        # get types where possible to help with list->tuple change
        pm.add_pass(PartialTypeInference, "performs partial type inference")
        # make const lists tuples
        pm.add_pass(TransformLiteralUnrollConstListToTuple, "switch const list for tuples")
        # recompute partial typemap following IR change
        pm.add_pass(PartialTypeInference, "performs partial type inference")
        # canonicalise loops - use our patched version
        pm.add_pass(
            NumbaCudaMlirIterLoopCanonicalization,
            "switch iter loops for range driven loops",
        )
        # rewrite consts
        pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
        # do the unroll - we patched the module-level literal_unroll above,
        # so the standard MixedContainerUnroller will work
        pm.add_pass(MixedContainerUnroller, "performs mixed container unroll")
        # rewrite dynamic getitem to static getitem as it's possible some more
        # getitems will now be statically resolvable
        pm.add_pass(GenericRewrites, "Generic Rewrites")
        pm.add_pass(RewriteSemanticConstants, "rewrite semantic constants")
        pm.finalize()
        pm.run(state)
        return True


@register_pass(mutates_CFG=True, analysis_only=False)
class NumbaCudaMlirIterLoopCanonicalization(IterLoopCanonicalization):
    """
    numba_cuda_mlir-specific version of IterLoopCanonicalization that accepts both
    numba.misc.special.literal_unroll and numba.cuda.misc.special.literal_unroll.
    """

    _name = "numba_cuda_mlir_iter_loop_canonicalisation"

    _accepted_calls = (literal_unroll,)


@register_pass(mutates_CFG=True, analysis_only=False)
class NumbaCudaMlirInlineInlinables(InlineInlinables):
    """InlineInlinables that skips self-recursive functions to avoid infinite inlining."""

    _name = "numba_cuda_mlir_inline_inlinables"

    def _do_work(self, state, work_list, block, i, expr, inline_worker):
        try:
            to_inline = state.func_ir.get_definition(expr.func)
            val = getattr(to_inline, "value", None)
            if val and hasattr(val, "py_func") and self._is_self_recursive(val.py_func):
                return False
        except Exception:
            pass
        return super()._do_work(state, work_list, block, i, expr, inline_worker)

    @staticmethod
    def _is_self_recursive(pyfunc):
        """Check if a function's bytecode references its own name."""
        import dis

        for instr in dis.get_instructions(pyfunc):
            if instr.opname in ("LOAD_GLOBAL", "LOAD_DEREF") and instr.argval == pyfunc.__name__:
                return True
        return False


@register_pass(mutates_CFG=True, analysis_only=False)
class PostInlineWholeFunctionPlanners(FunctionPass):
    """Run extension planners after device-function inlining."""

    _name = "post_inline_whole_function_planners"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        return _planner_registry.apply(state)


@register_pass(mutates_CFG=True, analysis_only=False)
class ConstArgTypeFolding(FunctionPass):
    """Replace ``constargtype(x)`` with the Numba type of ``x`` and fold what depends on it.

    We partially type the function, replace each ``constargtype`` call whose
    argument has a type with a constant holding that type, and evaluate every
    attribute read, operator, subscript and call whose operands are all constant
    and include such a type. Then we prune the branches those constants decide,
    which can let more arguments type, so we repeat until no call is left.
    """

    _name = "const_arg_type_folding"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        func_ir = state.func_ir
        func_ir._definitions = build_definitions(func_ir.blocks)
        calls = self._calls(func_ir)
        if not calls:
            return False
        folded = set()
        while calls:
            typemap = self._partial_types(state)
            progress = False
            for assign in calls:
                argtype = typemap.get(assign.value.args[0].name)
                if argtype is None or argtype is types.unknown:
                    continue
                assign.value = ir.Const(types.unliteral(argtype), assign.value.loc)
                folded.add(assign.target.name)
                progress = True
            if not progress:
                raise errors.TypingError(
                    "Cannot determine the type of the argument to constargtype", loc=calls[0].loc
                )
            self._fold(func_ir, folded)
            func_ir._definitions = build_definitions(func_ir.blocks)
            dead_branch_prune(func_ir, state.args)
            func_ir._definitions = build_definitions(func_ir.blocks)
            calls = self._calls(func_ir)
        self._remove_unused(func_ir, folded)
        func_ir._definitions = build_definitions(func_ir.blocks)
        return True

    @staticmethod
    def _is_constargtype(func_ir, var):
        from numba_cuda_mlir.cuda.experimental import constargtype

        definition = guard(get_definition, func_ir, var)
        return isinstance(definition, (ir.Global, ir.FreeVar)) and definition.value is constargtype

    def _calls(self, func_ir):
        calls = []
        for block in func_ir.blocks.values():
            for assign in block.find_insts(ir.Assign):
                value = assign.value
                if (
                    isinstance(value, ir.Expr)
                    and value.op == "call"
                    and len(value.args) == 1
                    and not value.kws
                    and self._is_constargtype(func_ir, value.func)
                ):
                    calls.append(assign)
        return calls

    @staticmethod
    def _partial_types(state):
        from numba_cuda_mlir.numba_cuda.core.typed_passes import type_inference_stage

        typemap, _, _, _ = type_inference_stage(
            state.typingctx,
            state.targetctx,
            state.func_ir,
            state.args,
            None,
            state.locals,
            raise_errors=False,
        )
        return typemap

    @staticmethod
    def _fold(func_ir, folded):
        """Evaluate expressions over constants that include a folded type, until none is left."""
        values = {}

        def constant(var):
            if var.name in values:
                return True, values[var.name]
            definition = guard(get_definition, func_ir, var)
            if isinstance(definition, (ir.Const, ir.Global, ir.FreeVar)):
                return True, definition.value
            # Read attribute chains such as types.float32 on constants too.
            if isinstance(definition, ir.Expr) and definition.op == "getattr":
                ok, base = constant(definition.value)
                if ok:
                    try:
                        return True, getattr(base, definition.attr)
                    except AttributeError:
                        pass
            return False, None

        def evaluate(expr):
            operands = [var for var in expr.list_vars()]
            if not any(var.name in folded for var in operands):
                return False, None
            known = {}
            for var in operands:
                ok, value = constant(var)
                if not ok:
                    return False, None
                known[var.name] = value
            if expr.op == "getattr":
                return True, getattr(known[expr.value.name], expr.attr)
            if expr.op in ("binop", "inplace_binop"):
                return True, expr.fn(known[expr.lhs.name], known[expr.rhs.name])
            if expr.op == "unary":
                return True, expr.fn(known[expr.value.name])
            if expr.op == "static_getitem":
                return True, known[expr.value.name][expr.index]
            if expr.op == "getitem":
                return True, known[expr.value.name][known[expr.index.name]]
            if expr.op == "call" and not expr.kws and expr.vararg is None:
                function = known[expr.func.name]
                # Calling a jitted function here would run it on the host, and
                # dead_branch_prune reads a branch condition through its bool() call.
                if function is bool or getattr(function, "targetoptions", None) is not None:
                    return False, None
                return True, function(*(known[arg.name] for arg in expr.args))
            return False, None

        changed = True
        while changed:
            changed = False
            for block in func_ir.blocks.values():
                for assign in block.find_insts(ir.Assign):
                    if assign.target.name in folded:
                        if isinstance(assign.value, ir.Const):
                            values[assign.target.name] = assign.value.value
                        continue
                    if not isinstance(assign.value, ir.Expr):
                        continue
                    try:
                        ok, value = evaluate(assign.value)
                    except Exception:
                        continue
                    if ok:
                        assign.value = ir.Const(value, assign.value.loc)
                        values[assign.target.name] = value
                        folded.add(assign.target.name)
                        changed = True

    @staticmethod
    def _remove_unused(func_ir, folded):
        """Drop folded constants and constargtype globals that nothing uses, since they cannot be typed."""
        while True:
            counts = Counter()
            for block in func_ir.blocks.values():
                for stmt in block.body:
                    counts.update(var.name for var in stmt.list_vars())
                    if isinstance(stmt, ir.Assign):
                        counts[stmt.target.name] -= 1
            used = {name for name, count in counts.items() if count > 0}
            removed = False
            for block in func_ir.blocks.values():
                kept = []
                for stmt in block.body:
                    if (
                        isinstance(stmt, ir.Assign)
                        and stmt.target.name not in used
                        and (
                            stmt.target.name in folded
                            or ConstArgTypeFolding._is_constargtype(func_ir, stmt.target)
                        )
                    ):
                        removed = True
                        continue
                    kept.append(stmt)
                block.body = kept
            if not removed:
                return
            func_ir._definitions = build_definitions(func_ir.blocks)
