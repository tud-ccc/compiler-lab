"""Task 1: loop fusion on buffers.

Fuses producer/consumer loop nests (e.g. matmul -> bias add -> relu) and
removes the intermediate buffer. Must be conservative: only fuse when all
buffers involved are distinct `memref.alloc`s created in the function,
because function arguments might alias.
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class FuseLoopsPass(ModulePass):
    name = "algeco-fuse-loops"

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # TODO (Task 1)
        pass
