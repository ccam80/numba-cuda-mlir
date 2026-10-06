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
from numba_cuda_mlir.numba_cuda.core.inline_closurecall import (
    InlineWorker,
    callee_ir_validator,
    current_inline_caller,
)
from numba_cuda_mlir.numba_cuda.core.ir_utils import (
    compute_cfg_from_blocks,
    dead_code_elimination,
    simplify_CFG,
)
from numba_cuda_mlir.numba_cuda.misc.special import literal_unroll
from numba_cuda_mlir._whole_function_planners import _planner_registry


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
class InlineTypedInlinables(FunctionPass):
    """Inline calls to inlinable device functions into the typed caller.

    Type inference types each inlinable call from its callee, transformed and
    typed for the call's argument types, through the current InlineCaller. This
    pass splices that typed callee IR at each call site and updates the
    caller's typemap and calltypes. Inlined bodies are added to the work list,
    so calls inside them are inlined too.
    """

    _name = "inline_typed_inlinables"

    def __init__(self):
        FunctionPass.__init__(self)

    def run_pass(self, state):
        caller = current_inline_caller()
        if caller is None or not caller.inlinees:
            return False

        inline_worker = InlineWorker(
            state.typingctx,
            state.targetctx,
            state.locals,
            state.pipeline,
            state.flags,
            callee_ir_validator,
            state.typemap,
            state.calltypes,
        )
        modified = False
        work_list = list(state.func_ir.blocks.items())
        while work_list:
            _, block = work_list.pop()
            for i, instr in enumerate(block.body):
                if (
                    isinstance(instr, ir.Assign)
                    and isinstance(instr.value, ir.Expr)
                    and instr.value.op == "call"
                ):
                    if self._inline_call(state, caller, inline_worker, work_list, block, i):
                        modified = True
                        break  # the block was split

        if modified:
            cfg = compute_cfg_from_blocks(state.func_ir.blocks)
            for dead in cfg.dead_nodes():
                del state.func_ir.blocks[dead]
            dead_code_elimination(state.func_ir, typemap=state.typemap)
            state.func_ir.blocks = simplify_CFG(state.func_ir.blocks)
        return modified

    def _inline_call(self, state, caller, inline_worker, work_list, block, i):
        from numba_cuda_mlir.numba_cuda.compiler import run_frontend
        from numba_cuda_mlir.numba_cuda.core.options import InlineOptions

        expr = block.body[i].value
        dispatcher = getattr(state.typemap.get(expr.func.name), "dispatcher", None)
        signature = state.calltypes.get(expr)
        if dispatcher is None or signature is None:
            return False
        iinfo = caller.inlinee(dispatcher, signature.args)
        if iinfo is None:
            return False

        inline_type = dispatcher.targetoptions["inline"]
        if InlineOptions(inline_type).has_cost_model:
            if not inline_type(expr, state.func_ir, run_frontend(dispatcher.py_func)):
                # Lowering calls the compiled function in place of the inlined body.
                dispatcher._compile_as_device_callee(signature.args)
                return False

        freevars = iinfo.func_ir.func_id.func.__code__.co_freevars
        _, _, _, new_blocks = inline_worker.inline_ir(
            state.func_ir,
            block,
            i,
            iinfo.func_ir,
            freevars,
            arg_typs=signature.args,
        )
        work_list.extend(new_blocks)
        return True
