# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-unroll metadata for ``range`` loops with compile-time trip counts.

Lowering tags each such loop's header branch through a NameLoc; after the base
pipeline :func:`annotate_static_range_loops` moves the hint onto the loop's
latch terminators as ``#llvm.loop_annotation<unroll = <full = true>>``.
"""

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.mlir.util import find_ops

STATIC_RANGE_LOOP_LOC_PREFIX = "numba_cuda_mlir.range_loop:"

_LATCH_TERMINATORS = frozenset({"llvm.br", "llvm.cond_br"})


def static_range_loop_location(trip_count: int) -> ir.Location:
    """Header-branch tag for a static-trip ``range`` loop, wrapping the current location."""
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
    """Dominator sets over ``blocks`` (reachable, entry first) by iterative data-flow."""
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


def _annotate_function(func_op, max_trip_count: int) -> int:
    all_blocks = list(func_op.regions[0].blocks)
    if not all_blocks:
        return 0
    blocks = _reachable_blocks(all_blocks[0])

    headers: list[tuple[ir.Block, ir.Operation]] = []
    for block in blocks:
        term = _terminator(block)
        if term is None or term.name != "llvm.cond_br":
            continue
        trip_count = _find_static_range_tag(term.location)
        if trip_count is None:
            continue
        if trip_count > max_trip_count:
            term.location = _strip_static_range_tag(term.location)
            continue
        headers.append((block, term))
    if not headers:
        return 0

    dom = _dominators(blocks)
    unroll_full = ir.Attribute.parse("#llvm.loop_annotation<unroll = <full = true>>")

    annotated = 0
    for header, header_term in headers:
        # Latches: predecessors of the header that the header dominates.
        latch_terms = []
        for block in blocks:
            term = _terminator(block)
            if term is None or header not in dom[block]:
                continue
            if any(succ == header for succ in term.successors):
                latch_terms.append(term)
        # Every latch must take the annotation; only br/cond_br can carry it.
        if latch_terms and all(t.name in _LATCH_TERMINATORS for t in latch_terms):
            for term in latch_terms:
                term.attributes["loop_annotation"] = unroll_full
            annotated += 1
        header_term.location = _strip_static_range_tag(header_term.location)
    return annotated


def annotate_static_range_loops(module: ir.Module) -> int:
    """Annotate tagged loops up to config.CUDA_UNROLL_MAX_TRIP_COUNT trips; returns the count."""
    from numba_cuda_mlir.numba_cuda import config

    max_trip_count = int(config.CUDA_UNROLL_MAX_TRIP_COUNT)
    count = 0
    for func_op in find_ops(module, lambda o: o.name == "llvm.func"):
        count += _annotate_function(func_op, max_trip_count)
    return count
