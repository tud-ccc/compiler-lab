"""Task 1: strip-mining and tiling of perfect `scf.for` nests.

**Strip-mining** splits one loop into a loop over tiles and a loop within a tile.
The iterations run in the same order as before, so it is always legal::

    scf.for %i = %c0 to %c64 step %c1 {    scf.for %ii = %c0 to %c64 step %c32 {
      ... %i ...                        ->    scf.for %i0 = %c0 to %c32 step %c1 {
    }                                           %i = arith.addi %ii, %i0
                                                ... %i ...
                                            } }

The original loop variable `%i` is computed in the *innermost* body of the nest,
not in the body of `%i0`: every loop in between still contains exactly one loop,
so the nest stays perfect and `interchange` can rebuild it.

**Tiling** strip-mines every loop of a nest and moves the tile loops outside::

    i, j, k  ->  ii, i0, jj, j0, kk, k0  ->  ii, jj, kk, i0, j0, k0

The second step is a loop interchange. Moving the tile loops out is legal if
*every* order of the original loops is legal (the nest is fully permutable),
which `is_legal` from the interchange pass decides. The relative order of the
loops is kept, so run `algeco-interchange` first to get a good order.

Assumptions, in addition to those of interchange: loops start at 0 with step 1,
and a tile size divides the trip count of its loop. A loop whose trip count is
not a multiple of its tile size, or whose tile size is 0, 1 or the whole trip
count, is not strip-mined: no remainder loops. The tile sizes apply to every
nest of matching depth.

Usage::

    algeco input.mlir -p 'algeco-strip-mine{sizes=32,32,32}'
    algeco input.mlir -p 'algeco-interchange,algeco-tile{sizes=32,32,32}'
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations

from xdsl.context import Context
from xdsl.dialects import arith, builtin, scf  # noqa: F401  (for your code)
from xdsl.dialects.builtin import IndexType, IntegerAttr  # noqa: F401
from xdsl.ir import Block  # noqa: F401
from xdsl.passes import ModulePass
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint  # noqa: F401

from algeco.transforms.interchange import (
    constant_index,
    extract_accesses,
    interchange,  # noqa: F401
    is_legal,
    perfect_nest,
    trip_counts,
)


def strip_mined(size: int, trip: int) -> bool:
    """Whether a loop of `trip` iterations is strip-mined with tiles of `size`."""
    return 1 < size < trip and trip % size == 0


# ---------------------------------------------------------------------------
# 1. Strip-mining
# ---------------------------------------------------------------------------


def strip_mine(
    loop: scf.ForOp, innermost: scf.ForOp, size: int, rewriter: PatternRewriter
) -> tuple[scf.ForOp, scf.ForOp]:
    """Replace `loop` (from 0, step 1) by a tile loop and a point loop.

    Returns the two new loops, outer first. The point loop runs from 0 to
    `size`; the original loop variable becomes `tile + point`, computed at the
    start of the body of `innermost`, the innermost loop of the nest (`loop`
    itself, if it has no inner loop).
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# 2. Tiling = strip-mining + interchange
# ---------------------------------------------------------------------------


def tile(
    nest: Sequence[scf.ForOp], sizes: Sequence[int], rewriter: PatternRewriter
) -> None:
    """Strip-mine every loop of `nest` (see `strip_mined`), tile loops outside.

    The order of the loops is kept: `i, j, k` with all three strip-mined gives
    `ii, jj, kk, i0, j0, k0`. Legality is the caller's business.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# The passes
# ---------------------------------------------------------------------------


@dataclass
class TileNest(RewritePattern):
    """Strip-mines, or tiles, every outermost perfect nest of depth `len(sizes)`."""

    sizes: tuple[int, ...]
    tile: bool  # False: only strip-mine

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: scf.ForOp, rewriter: PatternRewriter) -> None:
        if isinstance(op.parent_op(), scf.ForOp):
            return  # only start at the outermost loop of a nest
        nest = perfect_nest(op)
        if nest is None or len(nest) != len(self.sizes):
            return
        for loop in nest:
            if constant_index(loop.lb) != 0 or constant_index(loop.step) != 1:
                return
        accesses = extract_accesses(nest)
        if accesses is None:
            return  # accesses too complex for our model (e.g. already tiled)
        if not any(map(strip_mined, self.sizes, trip_counts(nest))):
            return  # nothing to do
        if self.tile:
            # Moving the tile loops out is an interchange: legal if every
            # order of the original loops is.
            loops = range(len(nest))
            if not all(is_legal(accesses, order) for order in permutations(loops)):
                return
            tile(nest, self.sizes, rewriter)
        else:
            innermost = nest[-1]
            trips = trip_counts(nest)
            for loop, size, trip in zip(nest, self.sizes, trips, strict=True):
                if strip_mined(size, trip):
                    _, inner = strip_mine(loop, innermost, size, rewriter)
                    if loop is innermost:
                        innermost = inner


@dataclass(frozen=True)
class StripMinePass(ModulePass):
    name = "algeco-strip-mine"

    # One tile size per loop, outermost first: 'algeco-strip-mine{sizes=32,32,8}'.
    sizes: tuple[int, ...] = (32, 32, 32)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            TileNest(self.sizes, tile=False), apply_recursively=False
        ).rewrite_module(op)


@dataclass(frozen=True)
class TileLoopsPass(ModulePass):
    name = "algeco-tile"

    # One tile size per loop, outermost first: 'algeco-tile{sizes=32,32,8}'.
    sizes: tuple[int, ...] = (32, 32, 32)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        PatternRewriteWalker(
            TileNest(self.sizes, tile=True), apply_recursively=False
        ).rewrite_module(op)
