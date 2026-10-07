"""Task 1: loop interchange.

Reorders the loops of a perfect `scf.for` nest. For the matmul emitted by the
frontend, `i, j, k -> i, k, j` makes the innermost loop walk `w[k, j]` and
`out[i, j]` contiguously instead of jumping a whole row of `w` per iteration.

The pass is split into four steps. Only the first and the last look at the IR;
the two decisions in between work on a small model of the nest:

1. **IR -> model** (`perfect_nest`, `trip_counts`, `extract_accesses`):
   number the loops 0, 1, 2, ... from the outside in, and describe every
   memory access by, for each index, the number of the loop whose variable is
   used there. A loop is named after its variable; for the matmul nest
   `i, j, k` (the `i` loop is loop 0, the `j` loop 1, the `k` loop 2):

       x[i, k] -> (0, 2)      w[k, j] -> (2, 1)      out[i, j] -> (0, 1)

2. **Legality** (`is_legal`): does the new order compute the same result?
3. **Profitability** (`choose_order`): which order touches the fewest cache lines?
4. **Model -> IR** (`interchange`): rebuild the nest in the new order.

Assumptions. Anything else is left unchanged:

- a nest is perfect: each loop body is exactly one inner loop (plus `scf.yield`),
  only the innermost body contains other operations
- loop bounds and steps are constants, and loops carry no values (`iter_args`)
- the innermost body only contains `memref.load`, `memref.store` and `arith` ops
- every index is a loop variable or a constant (no `i + 1`, no `2 * i`)
- every buffer is a function argument, a `memref.alloc` or a `memref.get_global`.
  A buffer that is written never overlaps another buffer: allocations are
  fresh, and algeco's calling convention guarantees that the output argument
  doesn't alias the inputs (see `algeco.jit`).

Usage::

    algeco input.mlir -p algeco-interchange                 # choose by cost model
    algeco input.mlir -p 'algeco-interchange{order=0,2,1}'  # force an order
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from xdsl.context import Context
from xdsl.dialects import arith, builtin, func, memref, scf
from xdsl.dialects.builtin import IndexType, IntegerAttr
from xdsl.ir import Block, BlockArgument, SSAValue
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Access:
    """One `memref.load` or `memref.store` in the innermost loop body.

    `buffer` identifies the buffer (any hashable value; in the pass it is the
    SSA value of the memref). `dims` has one entry per index: the number of the
    loop whose variable is used as that index, or None for a constant index.
    """

    buffer: object
    dims: tuple[int | None, ...]
    is_write: bool


# ---------------------------------------------------------------------------
# 1. IR -> model
# ---------------------------------------------------------------------------


def constant_index(value: SSAValue) -> int | None:
    """The value of an `arith.constant` of type index, or None if it isn't one."""
    op = value.owner
    if isinstance(op, arith.ConstantOp) and isinstance(op.value, IntegerAttr):
        return op.value.value.data
    return None


def perfect_nest(outer: scf.ForOp) -> list[scf.ForOp] | None:
    """The loops of the perfect nest starting at `outer`, from the outside in.

    Follow the chain of loops as long as a loop body is exactly
    `[scf.for, scf.yield]`. Return None if any loop in the chain carries values
    or has non-constant bounds or step, or a non-positive step.
    """
    nest = [outer]
    while True:
        body = list(nest[-1].body.block.ops)
        if len(body) == 2 and isinstance(body[0], scf.ForOp):
            nest.append(body[0])
        else:
            break
    for loop in nest:
        if loop.iter_args:
            return None
        bounds = [constant_index(v) for v in (loop.lb, loop.ub, loop.step)]
        if None in bounds or bounds[2] <= 0:
            return None
    return nest


def trip_counts(nest: Sequence[scf.ForOp]) -> list[int]:
    """The number of iterations of each loop (bounds are constants, see above)."""
    counts = []
    for loop in nest:
        lb, ub, step = (constant_index(v) for v in (loop.lb, loop.ub, loop.step))
        counts.append(max(0, -(-(ub - lb) // step)))  # ceil((ub - lb) / step)
    return counts


def is_known_buffer(buffer: SSAValue) -> bool:
    """A function argument, a fresh allocation, or a constant global."""
    if isinstance(buffer, BlockArgument):
        return isinstance(buffer.owner.parent_op(), func.FuncOp)
    return isinstance(buffer.owner, memref.AllocOp | memref.GetGlobalOp)


def extract_accesses(nest: Sequence[scf.ForOp]) -> list[Access] | None:
    """Describe every load and store of the innermost body as an `Access`.

    Return None (= don't touch this nest) if the body contains anything the
    model can't describe: an operation other than load, store, `arith` or
    `scf.yield`, an index that is neither a loop variable nor a constant, or a
    buffer that `is_known_buffer` rejects.
    """
    ivs = [loop.body.block.args[0] for loop in nest]
    accesses = []
    for op in nest[-1].body.block.ops:
        if isinstance(op, memref.LoadOp | memref.StoreOp):
            if not is_known_buffer(op.memref):
                return None
            dims: list[int | None] = []
            for index in op.indices:
                if index in ivs:
                    dims.append(ivs.index(index))
                elif constant_index(index) is not None:
                    dims.append(None)
                else:
                    return None
            accesses.append(
                Access(op.memref, tuple(dims), isinstance(op, memref.StoreOp))
            )
        elif not (isinstance(op, scf.YieldOp) or op.name.startswith("arith.")):
            return None
    return accesses


# ---------------------------------------------------------------------------
# 2. Legality
# ---------------------------------------------------------------------------


def is_legal(accesses: Sequence[Access], order: Sequence[int]) -> bool:
    """Can the loops run in `order` (new position -> old loop number)?

    Reading the same memory in a different order is harmless; only buffers that
    are written matter. A conservative rule that covers the frontend's loops:

    - every access to a written buffer uses the same `dims`, without constants.
      Then two iterations touch the same element exactly when they have the same
      values of the loop variables in `dims`, and differ only in the others.
    - the loops whose variables are not in `dims` (the reduction loops, like the
      `k` loop for `out[i, j] += ...`) keep their relative order. Then every
      element sees the same sequence of updates as before, so even
      floating-point results stay bit-identical.

    Everything else (e.g. `a[i, j] = a[j, i]`) is rejected.
    """
    written = {a.buffer for a in accesses if a.is_write}
    for buffer in written:
        all_dims = {a.dims for a in accesses if a.buffer == buffer}
        if len(all_dims) != 1:
            return False
        (dims,) = all_dims
        if None in dims:
            return False
        reduction_loops = [loop for loop in range(len(order)) if loop not in dims]
        new_positions = [order.index(loop) for loop in reduction_loops]
        if new_positions != sorted(new_positions):
            return False
    return True


# ---------------------------------------------------------------------------
# 3. Profitability
# ---------------------------------------------------------------------------

FLOATS_PER_CACHE_LINE = 16  # 64-byte cache lines, float32


def choose_order(accesses: Sequence[Access], trips: Sequence[int]) -> tuple[int, ...]:
    """The loop order with the cheapest loops innermost (new position -> old loop).

    Sorting is stable, so loops with equal cost keep their current order, and
    a nest that is already in the best order gets the identity.
    """
    pass


# ---------------------------------------------------------------------------
# 4. Model -> IR
# ---------------------------------------------------------------------------


def interchange(
    nest: Sequence[scf.ForOp], order: Sequence[int], rewriter: PatternRewriter
) -> None:
    """Replace `nest` by the same loops in `order` (new position -> old loop).

    Build a new nest whose loop at position `d` has the bounds and step of
    `nest[order[d]]`, move the innermost body into it, and make every use of an
    old loop variable use the corresponding new one.
    """
    pass


# ---------------------------------------------------------------------------
# The pass
# ---------------------------------------------------------------------------


@dataclass
class InterchangeNest(RewritePattern):
    """Applies steps 1-4 to every outermost `scf.for`."""

    forced_order: tuple[int, ...] = ()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        if isinstance(op.parent_op(), scf.ForOp):
            return  # Only start at the outermost loop of a nest
        nest = perfect_nest(op)
        if nest is None or len(nest) < 2:
            return # Not a perfect loop nest.
        accesses = extract_accesses(nest)
        if accesses is None:
            return # Accesses too complex for our logic
        if self.forced_order:
            order = self.forced_order
            if sorted(order) != list(range(len(nest))):
                return  # not a permutation of this nest's loops
        else:
            # Pick a profitable and legal order.
            order = choose_order(accesses, trip_counts(nest))

        if list(order) == list(range(len(nest))) or not is_legal(accesses, order):
            return # No change needed, or illegal

        # Apply the interchange
        interchange(nest, order, rewriter)


@dataclass(frozen=True)
class InterchangePass(ModulePass):
    name = "algeco-interchange"

    # Force this order on every nest of matching depth instead of using the
    # cost model, e.g. 'algeco-interchange{order=0,2,1}'. Still checked for legality.
    order: tuple[int, ...] = ()

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # One sweep: with a forced order, a second sweep would permute again.
        PatternRewriteWalker(
            InterchangeNest(self.order), apply_recursively=False
        ).rewrite_module(op)
