"""algeco: the algebra compiler of the Compilerbau Lab.

Pipeline overview::

    frontend: expression tree (Python operators on placeholders)
        -> loops on buffers (scf + memref + arith)     LoopEmitter
        -> your transformations: interchange, tiling, ... (xDSL passes)
        -> LLVM IR                                     mlir-opt / mlir-translate
        -> machine code                                clang (native or RISC-V)
"""

from algeco.frontend import broadcast_to, jit, matmul, maximum, relu

__version__ = "0.1.0"
__all__ = ["broadcast_to", "jit", "matmul", "maximum", "relu"]
