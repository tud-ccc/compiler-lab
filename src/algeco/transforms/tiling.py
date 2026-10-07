"""Task 1: loop tiling (strip-mining + interchange) of `scf.for` nests."""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class TileLoopsPass(ModulePass):
    name = "algeco-tile"

    # Usage: algeco -p 'algeco-tile{tile-sizes=32,32,8}'
    tile_sizes: tuple[int, ...] = (32, 32, 8)

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # TODO (Task 1)
        pass
