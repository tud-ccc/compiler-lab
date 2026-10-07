"""Task 3: vectorize innermost loops using the `vector` dialect.

Targets: native (x86-64 AVX / AArch64 NEON) and RISC-V (RVV).
"""

from dataclasses import dataclass

from xdsl.context import Context
from xdsl.dialects import builtin
from xdsl.passes import ModulePass


@dataclass(frozen=True)
class VectorizeLoopsPass(ModulePass):
    name = "algeco-vectorize"

    vector_width: int = 8

    def apply(self, ctx: Context, op: builtin.ModuleOp) -> None:
        # TODO (Task 3)
        pass
