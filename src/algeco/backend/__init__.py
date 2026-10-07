"""Backend: hand the middle-end output to upstream MLIR and LLVM tools."""

from algeco.backend.llvm import build_shared_library, compile_llvm_ir
from algeco.backend.mlir import (
    check_lowerable,
    lower_to_llvm_dialect,
    lower_to_llvm_ir,
    translate_to_llvm_ir,
)
from algeco.backend.runtime import CompiledFunction

__all__ = [
    "CompiledFunction",
    "build_shared_library",
    "check_lowerable",
    "compile_llvm_ir",
    "lower_to_llvm_dialect",
    "lower_to_llvm_ir",
    "translate_to_llvm_ir",
]
