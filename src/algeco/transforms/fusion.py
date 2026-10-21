"""Task 1: loop fusion on buffers.

Fuses two consecutive loop nests into one, and removes intermediate buffers
that nobody reads any more. The broadcast of the bias into the add::

    scf.for %i ... { scf.for %j ... {          scf.for %i ... { scf.for %j ... {
      %v = memref.load %b[%j]                    %v = memref.load %b[%j]
      memref.store %v, %t1[%i, %j] } }     ->    %a = memref.load %t0[%i, %j]
    scf.for %i ... { scf.for %j ... {            %s = arith.addf %a, %v
      %a = memref.load %t0[%i, %j]               memref.store %s, %t2[%i, %j] } }
      %w = memref.load %t1[%i, %j]
      %s = arith.addf %a, %w
      memref.store %s, %t2[%i, %j] } }

Two nests are fused at the depth of their *common prefix*: the outermost loops
with the same bounds and step. The consumer's body at that depth is appended
to the producer's. The matmul nest `i, j, k` and the add `i, j` share `i, j`::

    scf.for %i { scf.for %j {
      memref.store %zero, %t0[%i, %j]          // from the zero-fill nest
      scf.for %k { ... %t0[%i, %j] ... }       // the matmul
      ... %t2[%i, %j] = %t0[%i, %j] + %b[%j]   // the add, with the bias forwarded
    } }

**Legality.** Within the fused body, the producer's statements run before the
consumer's, so a dependence between the two nests is preserved exactly when it
goes from an iteration of the common loops to the same or a later one. We only
fuse if every buffer that one nest writes and the other accesses uses the
common loop variables at the same index positions in both nests, and all of
them (`%t0[%i, %j]` in both, not `%t0[%j, %i]` in one; `%t[%i, %k]` and
`%t[%i, %j]` are fine at depth 1). Two accesses then only touch the same
element in the same iteration of the common loops: iteration (i, j) of the
consumer reads what iteration (i, j) of the producer wrote. Distinct buffers
are assumed not to overlap (fresh allocations, and the calling convention of
`algeco.jit`); a compiler without that guarantee needs alias analysis here.

After fusing, a load of a value that was just stored into the same element is
replaced by the value (store-to-load forwarding). A buffer that is then never
read is removed, with its stores: `is_dead_intermediate` decides when that is
allowed.

Task 1: `can_fuse` (legality, on the model), `fuse` (the rewrite),
`forward_stores` and `is_dead_intermediate` are yours. The driver
(`fuse_next`, `fuse_block`), the model (`nest_accesses`) and the buffer
removal are provided. Until `can_fuse` says yes, the pass changes nothing.

Usage::

    algeco input.mlir -p algeco-fuse-loops
"""

from collections.abc import Sequence
from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.ir import Block, Operation, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

from algeco.transforms.interchange import (
    Access,
    constant_index,
    is_known_buffer,
)

# ---------------------------------------------------------------------------
# 1. IR -> model
# ---------------------------------------------------------------------------


def loop_chain(outer: scf.ForOp) -> list[scf.ForOp] | None:
    """The loops nested directly in each other from `outer`, outside in.

    Unlike `perfect_nest`, a body may contain other operations next to its
    inner loop (a fused nest). Stops at the first body with several loops.
    """
    nest = [outer]
    while True:
        inner = [op for op in nest[-1].body.block.ops if isinstance(op, scf.ForOp)]
        if len(inner) != 1:
            return nest
        nest.append(inner[0])


def common_prefix(a: Sequence[scf.ForOp], b: Sequence[scf.ForOp]) -> int:
    """How many loops, from the outside, have the same constant bounds and step."""
    depth = 0
    for x, y in zip(a, b, strict=False):
        if x.iter_args or y.iter_args:
            break
        bounds_x = [constant_index(v) for v in (x.lb, x.ub, x.step)]
        bounds_y = [constant_index(v) for v in (y.lb, y.ub, y.step)]
        if None in bounds_x or bounds_x != bounds_y:
            break
        depth += 1
    return depth


def nest_accesses(outer: scf.ForOp) -> list[Access] | None:
    """Every load and store anywhere under `outer`, as `Access`es.

    `dims` holds the *depth* of the loop whose variable is used as each index
    (0 for `outer`), or None for an index that is the same in every iteration:
    a constant, or a value defined outside the nest, such as the variable of
    an enclosing loop. Returns None if the nest contains anything else than
    loops, loads, stores and `arith` operations, an index computed inside the
    nest, or an unknown buffer.
    """
    depth_of: dict[SSAValue, int] = {}
    accesses: list[Access] = []

    def defined_outside(value: SSAValue) -> bool:
        owner = value.owner
        op = owner if isinstance(owner, Operation) else owner.parent_op()
        while op is not None:
            if op is outer:
                return False
            op = op.parent_op()
        return True

    def visit(loop: scf.ForOp, depth: int) -> bool:
        depth_of[loop.body.block.args[0]] = depth
        for op in loop.body.block.ops:
            if isinstance(op, scf.ForOp):
                if not visit(op, depth + 1):
                    return False
            elif isinstance(op, memref.LoadOp | memref.StoreOp):
                if not is_known_buffer(op.memref):
                    return False
                dims: list[int | None] = []
                for index in op.indices:
                    if index in depth_of:
                        dims.append(depth_of[index])
                    elif defined_outside(index):
                        dims.append(None)
                    else:
                        return False
                accesses.append(
                    Access(op.memref, tuple(dims), isinstance(op, memref.StoreOp))
                )
            elif not (isinstance(op, scf.YieldOp) or op.name.startswith("arith.")):
                return False
        return True

    return accesses if visit(outer, 0) else None


# ---------------------------------------------------------------------------
# 2. Legality
# ---------------------------------------------------------------------------


def common_loops(dims: Sequence[int | None], depth: int) -> tuple[int | None, ...]:
    """Where the first `depth` loops index an access: `(None, 1)` for `t[k, j]`."""
    return tuple(d if d is not None and d < depth else None for d in dims)


def can_fuse(
    producer: Sequence[Access], consumer: Sequence[Access], depth: int
) -> bool:
    """Can two nests be fused at `depth` without changing any dependence?

    Every buffer that one nest writes and the other nest accesses must use
    the common loops at the same index positions in all its accesses, and use
    all of them. Then two accesses to the same element always come from the
    same iteration of the common loops. Buffers that both nests only read, or
    that only one nest touches, don't matter.

    `common_loops` reduces an access to the positions of the common loops:
    all accesses to such a buffer must reduce to the same pattern, and the
    pattern must mention every loop below `depth`.
    """
    return False


def is_dead_intermediate(buffer: SSAValue) -> bool:
    """May `buffer` be removed, together with the stores into it?

    Only if nobody can observe its contents any more: it must be a
    `memref.alloc` of this function (arguments belong to the caller, globals
    are read-only), and its remaining uses (`buffer.uses`, each with an
    `.operation`) must be stores into it and its dealloc.
    """
    return False


# ---------------------------------------------------------------------------
# 3. Model -> IR
# ---------------------------------------------------------------------------


def fuse(
    producer: Sequence[scf.ForOp],
    consumer: Sequence[scf.ForOp],
    depth: int,
    rewriter: PatternRewriter,
) -> None:
    """Append the consumer's body at `depth` to the producer's; erase the consumer.

    `producer` and `consumer` are loop chains (outside in); their first `depth`
    loops are the common ones. Every use of a consumer loop variable becomes a
    use of the producer's; the operations of the consumer's body at `depth`
    (all but its `scf.yield`) move before the producer's `scf.yield` at that
    depth; then the consumer's outermost loop is erased.
    """
    raise NotImplementedError


def forward_stores(block: Block, rewriter: PatternRewriter) -> None:
    """Replace a load by the value stored into the same element just before.

    Walk the operations of `block` in order and remember, per buffer, the
    last store and its indices. A load with the same buffer and the same
    index values (the same SSA values) is replaced by the stored value:
    `rewriter.replace(load, [], [value])`. Only within `block`: a later store
    into the buffer replaces what we remember about it, and a nested loop
    (which may store anything) forgets everything.
    """
    pass


def remove_dead_buffers(nest: scf.ForOp, rewriter: PatternRewriter) -> None:
    """Remove the buffers stored under `nest` that `is_dead_intermediate` allows."""
    buffers = {op.memref for op in nest.walk() if isinstance(op, memref.StoreOp)}
    for buffer in buffers:
        if is_dead_intermediate(buffer):
            for use in list(buffer.uses):
                rewriter.erase(use.operation)
            if isinstance(buffer.owner, Operation):
                rewriter.erase(buffer.owner)


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


def next_nest(nest: scf.ForOp) -> tuple[scf.ForOp, list[Operation]] | None:
    """The next loop nest after `nest`, and the allocations between them.

    Anything else in between (a store, a copy, ...) is a reason not to fuse.
    """
    between: list[Operation] = []
    op = nest.next_op
    while op is not None:
        if isinstance(op, scf.ForOp):
            return op, between
        if not isinstance(op, memref.AllocOp | arith.ConstantOp):
            return None
        between.append(op)
        op = op.next_op
    return None


def fuse_next(op: scf.ForOp, rewriter: PatternRewriter) -> bool:
    """Fuse the nest after `op` into `op`, if that is legal. True if it was."""
    found = next_nest(op)
    if found is None:
        return False
    consumer_op, between = found
    producer, consumer = loop_chain(op), loop_chain(consumer_op)
    depth = common_prefix(producer, consumer)
    if depth == 0:
        return False
    p_accesses, c_accesses = nest_accesses(op), nest_accesses(consumer_op)
    if p_accesses is None or c_accesses is None:
        return False
    if not can_fuse(p_accesses, c_accesses, depth):
        return False

    # The consumer's buffers must be allocated before the fused nest.
    for alloc in between:
        alloc.detach()
        rewriter.insert(alloc, InsertPoint.before(op))
    fuse(producer, consumer, depth, rewriter)
    forward_stores(producer[depth - 1].body.block, rewriter)
    remove_dead_buffers(op, rewriter)
    return True


def fuse_block(block: Block, rewriter: PatternRewriter) -> None:
    """Fuse consecutive nests in `block`, then inside every loop of it.

    Fusing two layers on their batch loop puts the nests of the second layer
    next to each other in the batch loop's body, where the recursion fuses them.
    """
    op = block.first_op
    while op is not None:
        if isinstance(op, scf.ForOp):
            while fuse_next(op, rewriter):
                pass
            fuse_block(op.body.block, rewriter)
        op = op.next_op


@dataclass(frozen=True)
class FuseLoopsPass(ModulePass):
    name = "algeco-fuse-loops"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        for function in op.ops:
            if isinstance(function, func.FuncOp):
                fuse_block(function.body.block, PatternRewriter(function))
