# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intra-block scheduling of typed Numba IR before MLIR lowering.

The scheduler reorders the statements of each basic block without
changing the CFG, the SSA names, or the set of statements.  Legality is
expressed as a per-block dependency DAG:

- flow edges from each in-block definition to its in-block uses;
- a sequential chain over every statement touching a multiply-defined
  name (non-SSA names keep their original relative order);
- per-alias-root memory chains: stores to a root are kept in order,
  loads never cross a store to the same root, and statements whose
  alias root is unknown act as members of every root's chain;
- effectful or unrecognised statements are scheduling barriers;
- ``ir.Del`` statements stay after every prior statement that
  references their variable;
- leading ``Arg`` assignments are pinned to the block head, and blocks
  with phi assignments beyond that prefix are left untouched.

Any topological order of that DAG is a legal statement order.  The
``policy`` chooses which one to emit:

- ``"source"``: keep the original order (identity control);
- ``"dfs"``: roots-first predecessor postorder — each externally
  consumed value is emitted together with the full chain that computes
  it, mirroring CuBIE's ``liveness_auto`` source ordering;
- ``"liveness"``: greedy list schedule that at every step prefers the
  ready statement closing the most live values;
- ``"longlived_dfs"``: the ``"dfs"`` emission with roots reordered so
  chains feeding block-live-out (long-lived) values are emitted first
  and one-shot chains stay serial and late.
"""

import heapq
import os

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import ir
from numba_cuda_mlir._whole_function_planners import TypedWholeFunctionPlanner

_METADATA_KEY = "typed_block_scheduler"

_PURE_EXPR_OPS = frozenset(
    {
        "binop",
        "inplace_binop",
        "unary",
        "cast",
        "exhaust_iter",
        "build_tuple",
        "getattr",
        "null",
        "undef",
    }
)

_VIEW_EXPR_OPS = frozenset({"getitem", "static_getitem", "cast"})

_PURE_CALL_MODULE_ROOTS = frozenset(
    {"math", "cmath", "builtins", "operator", "numpy", "numba", "numba_cuda_mlir", "cubie"}
)

_IMPURE_CALL_MARKERS = (
    "sync",
    "atomic",
    "fence",
    "print",
    "random",
    "stwt",
    "vote",
    "shfl",
    "ballot",
    "match_any",
    "match_all",
)

_BARRIER_STATEMENT_TYPES = tuple(
    statement_type
    for statement_type in (
        getattr(ir, "Print", None),
        getattr(ir, "SetAttr", None),
        getattr(ir, "EnterWith", None),
        getattr(ir, "PopBlock", None),
        getattr(ir, "Raise", None),
        getattr(ir, "StaticRaise", None),
        getattr(ir, "DynamicRaise", None),
    )
    if statement_type is not None
)


def _expr_op(statement):
    if isinstance(statement, ir.Assign) and isinstance(statement.value, ir.Expr):
        return statement.value.op
    return None


class _Node:
    __slots__ = (
        "index",
        "statement",
        "defs",
        "uses",
        "successors",
        "predecessors",
    )

    def __init__(self, index, statement, defs, uses):
        self.index = index
        self.statement = statement
        self.defs = defs
        self.uses = uses
        self.successors = set()
        self.predecessors = set()


class TypedBlockScheduler(TypedWholeFunctionPlanner):
    """Reorder statements inside each block of the typed IR."""

    #: Ordering policy applied to every block; see the module docstring.
    policy = os.environ.get("NUMBA_CUDA_MLIR_BLOCK_SCHEDULE", "dfs")

    def run(self) -> bool:
        policy = self.state.metadata.get(
            "typed_block_scheduler_policy", type(self).policy
        )
        if policy not in {"source", "dfs", "liveness", "longlived_dfs"}:
            raise ValueError(f"unknown block schedule policy {policy!r}")
        func_ir = self.state.func_ir
        typemap = self.state.typemap
        roots = self._alias_roots(func_ir, typemap)
        live_out = self._block_live_out(func_ir)
        modified = False
        stats = {
            "blocks": 0,
            "reordered_blocks": 0,
            "moved_statements": 0,
            "statements": 0,
            "largest_block": 0,
        }
        for label, block in func_ir.blocks.items():
            stats["blocks"] += 1
            stats["statements"] += len(block.body)
            stats["largest_block"] = max(
                stats["largest_block"], len(block.body)
            )
            if policy == "source":
                continue
            order = self._schedule_block(
                block, policy, roots, typemap, live_out.get(label, frozenset())
            )
            if order is None:
                continue
            body = block.body
            new_body = [body[index] for index in order] + [body[-1]]
            moved = sum(
                1 for position, index in enumerate(order) if index != position
            )
            if moved:
                block.body = new_body
                stats["reordered_blocks"] += 1
                stats["moved_statements"] += moved
                modified = True
        stats["policy"] = policy
        self.state.metadata[_METADATA_KEY] = stats
        return modified

    # -- alias analysis ------------------------------------------------

    def _alias_roots(self, func_ir, typemap):
        """Map each array-typed name to a conservative allocation root.

        Roots are argument names, allocating calls, or the sentinel
        ``None`` for unknown provenance.  Views (``getitem``/
        ``static_getitem``/``cast`` returning an array) share their
        parent's root.
        """

        roots = {}

        def resolve(name, seen):
            if name in roots:
                return roots[name]
            if name in seen:
                return None
            seen.add(name)
            definitions = func_ir._definitions.get(name, [])
            if len(definitions) != 1:
                root = None
            else:
                value = definitions[0]
                if isinstance(value, ir.Arg):
                    root = ("arg", value.index)
                elif isinstance(value, ir.Var):
                    root = resolve(value.name, seen)
                elif isinstance(value, ir.Expr):
                    if value.op in _VIEW_EXPR_OPS:
                        parent = value.value
                        root = (
                            resolve(parent.name, seen)
                            if isinstance(parent, ir.Var)
                            else None
                        )
                    elif value.op == "call":
                        root = ("alloc", id(value))
                    else:
                        root = None
                else:
                    root = None
            roots[name] = root
            return root

        for name, numba_type in typemap.items():
            if isinstance(numba_type, types.Array):
                resolve(name, set())
        return roots

    # -- liveness ------------------------------------------------------

    def _block_live_out(self, func_ir):
        """Names referenced by any other block or a terminator, per block.

        A backward dataflow fixpoint is unnecessary for scheduling
        priorities: a value defined here and referenced anywhere else
        outlives this block, which is the only distinction the
        policies consume.
        """

        blocks_referencing = {}
        block_names = {}
        for label, block in func_ir.blocks.items():
            names = set()
            for statement in block.body:
                for var in statement.list_vars():
                    names.add(var.name)
            block_names[label] = names
            for name in names:
                blocks_referencing[name] = blocks_referencing.get(name, 0) + 1
        live_out = {}
        for label, block in func_ir.blocks.items():
            names = block_names[label]
            external = {
                name
                for name in names
                if blocks_referencing[name] > 1
            }
            external |= {var.name for var in block.terminator.list_vars()}
            live_out[label] = frozenset(external)
        return live_out

    # -- statement classification --------------------------------------

    def _resolve_callee(self, func_ir, expr):
        """Return the Python object a call expression targets, or None."""

        attributes = []
        value = expr.func
        for _ in range(8):
            try:
                definition = func_ir.get_definition(value)
            except Exception:
                return None
            if isinstance(definition, (ir.Global, ir.FreeVar)):
                target = definition.value
                for attribute in reversed(attributes):
                    try:
                        target = getattr(target, attribute)
                    except AttributeError:
                        return None
                return target
            if isinstance(definition, ir.Expr) and definition.op == "getattr":
                attributes.append(definition.attr)
                value = definition.value
                continue
            return None
        return None

    def _call_is_pure(self, func_ir, typemap, expr):
        """Whether a call expression is safe to reorder freely."""

        for argument in expr.list_vars():
            if argument is expr.func:
                continue
            if isinstance(typemap.get(argument.name), types.Array):
                return False
        callee = self._resolve_callee(func_ir, expr)
        if callee is None:
            return False
        qualname = (
            getattr(callee, "__qualname__", "")
            or getattr(callee, "__name__", "")
            or repr(callee)
        ).lower()
        module = getattr(callee, "__module__", "") or ""
        haystack = f"{module}.{qualname}"
        if any(marker in haystack for marker in _IMPURE_CALL_MARKERS):
            return False
        return module.split(".")[0] in _PURE_CALL_MODULE_ROOTS

    # -- graph construction --------------------------------------------

    def _schedule_block(self, block, policy, roots, typemap, live_out):
        body = block.body
        if len(body) < 3:
            return None
        statements = body[:-1]

        pinned = 0
        for statement in statements:
            if isinstance(statement, ir.Assign) and isinstance(
                statement.value, ir.Arg
            ):
                pinned += 1
                continue
            break
        movable = statements[pinned:]
        if len(movable) < 2:
            return None
        if any(_expr_op(statement) == "phi" for statement in movable):
            return None

        nodes = []
        for offset, statement in enumerate(movable):
            defs = set()
            uses = set()
            if isinstance(statement, ir.Assign):
                defs.add(statement.target.name)
                value = statement.value
                if isinstance(value, ir.Expr):
                    uses.update(var.name for var in value.list_vars())
                elif isinstance(value, ir.Var):
                    uses.add(value.name)
            elif isinstance(statement, ir.Del):
                uses.add(statement.value)
            else:
                uses.update(var.name for var in statement.list_vars())
            nodes.append(_Node(offset, statement, defs, uses))

        def add_edge(before, after):
            if before is after:
                return
            if after.index not in before.successors:
                before.successors.add(after.index)
                after.predecessors.add(before.index)

        # Flow edges and non-SSA chains.
        def_site = {}
        multi_def = set()
        for node in nodes:
            for name in node.defs:
                if name in def_site:
                    multi_def.add(name)
                def_site.setdefault(name, node)
        for node in nodes:
            for name in node.uses:
                site = def_site.get(name)
                if site is not None and site.index < node.index:
                    add_edge(site, node)
        touching = {}
        for node in nodes:
            for name in node.defs | node.uses:
                if name in multi_def:
                    previous = touching.get(name)
                    if previous is not None:
                        add_edge(previous, node)
                    touching[name] = node

        # Del statements follow everything that referenced their name.
        references = {}
        for node in nodes:
            if isinstance(node.statement, ir.Del):
                name = node.statement.value
                for other_index in references.get(name, ()):
                    add_edge(nodes[other_index], node)
            for name in node.defs | node.uses:
                references.setdefault(name, []).append(node.index)

        # Memory chains and barriers.
        func_ir = self.state.func_ir
        last_store = {}
        loads_since_store = {}
        last_barrier = None
        memory_nodes = []

        for node in nodes:
            statement = node.statement
            kind = None
            root = None
            if isinstance(statement, (ir.SetItem, ir.StaticSetItem)):
                kind = "store"
                root = roots.get(statement.target.name)
            elif isinstance(statement, _BARRIER_STATEMENT_TYPES):
                kind = "barrier"
            elif isinstance(statement, ir.Assign):
                op = _expr_op(statement)
                if op in ("getitem", "static_getitem", "typed_getitem"):
                    if isinstance(
                        typemap.get(statement.target.name), types.Array
                    ):
                        kind = None  # view creation is address arithmetic
                    else:
                        kind = "load"
                        root = roots.get(statement.value.value.name)
                elif op == "call":
                    if self._call_is_pure(func_ir, typemap, statement.value):
                        kind = None
                    else:
                        kind = "barrier"
                elif op in _PURE_EXPR_OPS or op is None or op == "phi":
                    kind = None
                else:
                    kind = "barrier"
            elif isinstance(statement, ir.Del):
                # Deleting an array ends its storage lifetime: pin it
                # against every memory operation on the same root.
                if isinstance(typemap.get(statement.value), types.Array):
                    kind = "store"
                    root = roots.get(statement.value)
                else:
                    kind = None
            else:
                kind = "barrier"

            if kind is None:
                continue
            if kind == "barrier":
                for other in memory_nodes:
                    add_edge(other, node)
                last_barrier = node
                memory_nodes.append(node)
                continue
            if last_barrier is not None:
                add_edge(last_barrier, node)
            memory_nodes.append(node)
            if root is None:
                keys = set(last_store) | set(loads_since_store) | {None}
                if kind == "store":
                    for key in keys:
                        store = last_store.get(key)
                        if store is not None:
                            add_edge(store, node)
                        for load_index in loads_since_store.get(key, ()):
                            add_edge(nodes[load_index], node)
                        last_store[key] = node
                        loads_since_store[key] = []
                else:
                    for key in keys:
                        store = last_store.get(key)
                        if store is not None:
                            add_edge(store, node)
                    loads_since_store.setdefault(None, []).append(node.index)
                continue
            unknown_store = last_store.get(None)
            if unknown_store is not None:
                add_edge(unknown_store, node)
            if kind == "store":
                store = last_store.get(root)
                if store is not None:
                    add_edge(store, node)
                for load_index in loads_since_store.get(root, ()):
                    add_edge(nodes[load_index], node)
                for load_index in loads_since_store.get(None, ()):
                    add_edge(nodes[load_index], node)
                last_store[root] = node
                loads_since_store[root] = []
            else:
                store = last_store.get(root)
                if store is not None:
                    add_edge(store, node)
                loads_since_store.setdefault(root, []).append(node.index)

        order = self._order_nodes(nodes, policy, live_out)
        if order is None:
            return None
        return list(range(pinned)) + [index + pinned for index in order]

    # -- ordering policies ---------------------------------------------

    def _order_nodes(self, nodes, policy, live_out):
        if policy == "liveness":
            return self._order_liveness(nodes, live_out)
        return self._order_dfs(nodes, live_out, policy == "longlived_dfs")

    def _order_dfs(self, nodes, live_out, longlived_first):
        """Roots-first predecessor postorder.

        Every node without in-block successors is a root: its value (or
        effect) is only consumed outside the block.  Emitting each
        root's unscheduled predecessor cone in postorder keeps each
        chain serial, so one-shot temporaries die immediately.  With
        ``longlived_first`` the roots defining block-live-out names are
        emitted before effect-only roots, so long-lived values are
        computed early and the short chains run late and serial.
        """

        root_nodes = [node for node in nodes if not node.successors]
        if longlived_first:

            def root_key(node):
                defines_live_out = bool(node.defs & live_out)
                return (0 if defines_live_out else 1, node.index)

            root_nodes.sort(key=root_key)
        scheduled = [False] * len(nodes)
        order = []

        for root in root_nodes:
            if scheduled[root.index]:
                continue
            stack = [(root, None)]
            while stack:
                node, iterator = stack.pop()
                if iterator is None:
                    if scheduled[node.index]:
                        continue
                    iterator = iter(sorted(node.predecessors))
                advanced = False
                for predecessor_index in iterator:
                    if not scheduled[predecessor_index]:
                        stack.append((node, iterator))
                        stack.append((nodes[predecessor_index], None))
                        advanced = True
                        break
                if not advanced and not scheduled[node.index]:
                    scheduled[node.index] = True
                    order.append(node.index)
        if len(order) != len(nodes):
            return None
        return order

    def _order_liveness(self, nodes, live_out):
        """Greedy list schedule minimising the live-value count.

        At every step the ready statement with the best
        ``(values opened) - (values closed)`` balance runs first, with
        the original position as the tie-break.  Scores go stale as
        values die, so heap entries are lazily re-scored on pop.
        """

        remaining_uses = {}
        for node in nodes:
            for name in node.uses:
                remaining_uses[name] = remaining_uses.get(name, 0) + 1

        def score(node):
            closes = sum(
                1
                for name in node.uses
                if remaining_uses.get(name, 0) == 1 and name not in live_out
            )
            opens = sum(1 for name in node.defs if name not in live_out)
            return opens - closes

        pending = {node.index: len(node.predecessors) for node in nodes}
        ready = [
            (score(node), node.index)
            for node in nodes
            if pending[node.index] == 0
        ]
        heapq.heapify(ready)
        order = []
        scheduled = [False] * len(nodes)
        while ready:
            stale_score, index = heapq.heappop(ready)
            if scheduled[index]:
                continue
            node = nodes[index]
            current_score = score(node)
            if current_score != stale_score:
                heapq.heappush(ready, (current_score, index))
                continue
            scheduled[index] = True
            order.append(index)
            for name in node.uses:
                remaining_uses[name] -= 1
            for successor_index in node.successors:
                pending[successor_index] -= 1
                if pending[successor_index] == 0:
                    successor = nodes[successor_index]
                    heapq.heappush(
                        ready, (score(successor), successor_index)
                    )
        if len(order) != len(nodes):
            return None
        return order


__all__ = ["TypedBlockScheduler"]
