# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unroll hints for ``range`` loops with compile-time trip counts.

Numba lowers ``for i in range(...)`` into an unstructured CFG around an
in-memory range iterator, so by the time LLVM sees the loop there is no
``scf.for`` left to unroll on the MLIR side; the backend (libnvvm / nvJitLink)
makes the unroll decision from its own cost model. That cost model counts the
loop body *before* memory operations are scalarized, which means a body that
reads a shared- or global-memory operand looks larger than the same body
reading a local array (whose accesses SROA turns into registers early), and a
loop that fully unrolls with a local operand stays rolled with a shared one
(#8). Attaching ``llvm.loop.unroll.full`` to loops whose trip count is a
compile-time constant makes the decision independent of that estimate: LLVM
honours the hint the way it honours ``#pragma unroll`` in CUDA C++, subject
only to its pragma size cap.

The lowering tags the header branch of each such loop through its *location*
(``STATIC_RANGE_LOOP_LOC_PREFIX``), because locations survive dialect
conversion and the CFG canonicalizations that fold pass-through latch blocks,
whereas attributes on ``cf`` branches do not. After the base pass pipeline
:func:`annotate_static_range_loops` recovers the loops from the tagged
headers, puts ``#llvm.loop_annotation<unroll = <full = true>>`` on the
terminator of every latch (LLVM reads loop metadata from the latch
terminators), and strips the tags.
"""

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.mlir.util import find_ops

STATIC_RANGE_LOOP_LOC_PREFIX = "numba_cuda_mlir.range_loop:"

_LATCH_TERMINATORS = frozenset({"llvm.br", "llvm.cond_br"})


def static_range_loop_location(trip_count: int) -> ir.Location:
    """Location tagging the header branch of a ``range`` loop with a
    compile-time trip count, nested inside the current location."""
    name = f"{STATIC_RANGE_LOOP_LOC_PREFIX}{trip_count}"
    child = ir.Location.current
    if child is None:
        return ir.Location.name(name)
    return ir.Location.name(name, child)


def _find_static_range_tag(loc: ir.Location) -> int | None:
    """Return the tagged trip count if ``loc`` carries the header tag."""
    if isinstance(loc, ir.NameLoc):
        if loc.name_str.startswith(STATIC_RANGE_LOOP_LOC_PREFIX):
            return int(loc.name_str[len(STATIC_RANGE_LOOP_LOC_PREFIX) :])
        return _find_static_range_tag(loc.child_loc)
    if isinstance(loc, ir.FusedLoc):
        for nested in loc.locations:
            tag = _find_static_range_tag(nested)
            if tag is not None:
                return tag
    return None


def _strip_static_range_tag(loc: ir.Location) -> ir.Location:
    """Return ``loc`` with the header tag removed from its location tree."""
    if isinstance(loc, ir.NameLoc):
        if loc.name_str.startswith(STATIC_RANGE_LOOP_LOC_PREFIX):
            return loc.child_loc
        return ir.Location.name(loc.name_str, _strip_static_range_tag(loc.child_loc))
    if isinstance(loc, ir.FusedLoc):
        stripped = [_strip_static_range_tag(nested) for nested in loc.locations]
        if len(stripped) == 1:
            return stripped[0]
        return ir.Location.fused(stripped) if stripped else loc
    return loc


def _terminator(block: ir.Block) -> ir.Operation | None:
    ops = block.operations
    if len(ops) == 0:
        return None
    return ops[len(ops) - 1].operation


def _reachable_blocks(entry: ir.Block) -> list[ir.Block]:
    """Blocks reachable from ``entry``, entry first."""
    seen = {entry}
    order = [entry]
    worklist = [entry]
    while worklist:
        term = _terminator(worklist.pop())
        if term is None:
            continue
        for succ in term.successors:
            if succ not in seen:
                seen.add(succ)
                order.append(succ)
                worklist.append(succ)
    return order


def _dominators(blocks: list[ir.Block]) -> dict[ir.Block, set[ir.Block]]:
    """Dominator sets over ``blocks`` (all reachable from ``blocks[0]``),
    by iterative data-flow."""
    entry = blocks[0]
    preds: dict[ir.Block, list[ir.Block]] = {b: [] for b in blocks}
    for block in blocks:
        term = _terminator(block)
        if term is None:
            continue
        for succ in term.successors:
            preds[succ].append(block)

    everything = set(blocks)
    dom: dict[ir.Block, set[ir.Block]] = {b: set(everything) for b in blocks}
    dom[entry] = {entry}
    changed = True
    while changed:
        changed = False
        for block in blocks:
            if block == entry:
                continue
            pred_doms = [dom[p] for p in preds[block]]
            new = {block} | (set.intersection(*pred_doms) if pred_doms else set())
            if new != dom[block]:
                dom[block] = new
                changed = True
    return dom


def _annotate_function(func_op) -> int:
    all_blocks = list(func_op.regions[0].blocks)
    if not all_blocks:
        return 0
    blocks = _reachable_blocks(all_blocks[0])

    headers: list[tuple[ir.Block, ir.Operation]] = []
    for block in blocks:
        term = _terminator(block)
        if term is None or term.name != "llvm.cond_br":
            continue
        if _find_static_range_tag(term.location) is not None:
            headers.append((block, term))
    if not headers:
        return 0

    dom = _dominators(blocks)
    unroll_full = ir.Attribute.parse("#llvm.loop_annotation<unroll = <full = true>>")

    annotated = 0
    for header, header_term in headers:
        # A latch is a predecessor of the header that the header dominates,
        # i.e. the source of a back edge.
        latch_terms = []
        for block in blocks:
            term = _terminator(block)
            if term is None or header not in dom[block]:
                continue
            if any(succ == header for succ in term.successors):
                latch_terms.append(term)
        # LLVM reads loop metadata from every latch terminator and requires
        # them to agree; the LLVM dialect models it only on br/cond_br, so a
        # loop with any other latch terminator is left alone.
        if latch_terms and all(t.name in _LATCH_TERMINATORS for t in latch_terms):
            for term in latch_terms:
                term.attributes["loop_annotation"] = unroll_full
            annotated += 1
        header_term.location = _strip_static_range_tag(header_term.location)
    return annotated


def annotate_static_range_loops(module: ir.Module) -> int:
    """Attach full-unroll loop metadata to every tagged static-trip ``range``
    loop in ``module`` and strip the header tags. Returns the number of loops
    annotated."""
    count = 0
    for func_op in find_ops(module, lambda o: o.name == "llvm.func"):
        count += _annotate_function(func_op)
    return count
