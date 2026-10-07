"""Compiling LLVM IR to assembly, objects, or shared libraries with clang."""

import tempfile
from pathlib import Path
from typing import Literal

from algeco.backend.tools import run_tool

# Common targets for Task 3. RISC-V output can be inspected (asm) but not run
# natively; there is no qemu on conda-forge.
TARGETS = {
    "native": None,
    "x86-64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
    "riscv64": "riscv64-unknown-linux-gnu",
}

# Extra flags per target, e.g. enabling vector extensions.
TARGET_FLAGS = {
    "riscv64-unknown-linux-gnu": ["-march=rv64gcv"],
}


def _target_args(target: str | None) -> list[str]:
    triple = TARGETS.get(target, target) if target else None
    if triple is None:
        return ["-march=native"]
    return [f"--target={triple}", *TARGET_FLAGS.get(triple, [])]


# Disables LLVM's own loop and SLP vectorizers, so that measurements show what
# algeco's vectorization passes contribute.
NO_LLVM_VECTORIZE = ["-fno-vectorize", "-fno-slp-vectorize"]


def compile_llvm_ir(
    llvm_ir: str,
    kind: Literal["asm", "obj"] = "obj",
    target: str | None = None,
    opt_level: str = "-O2",
    llvm_vectorize: bool = False,
) -> str | bytes:
    flag = "-S" if kind == "asm" else "-c"
    vec_flags = [] if llvm_vectorize else NO_LLVM_VECTORIZE
    out = run_tool(
        "clang",
        ["-x", "ir", "-", flag, opt_level, *vec_flags, *_target_args(target)]
        + ["-o", "-"],
        llvm_ir,
    )
    return out.decode() if kind == "asm" else out


def build_shared_library(
    llvm_ir: str, out_dir: Path | None = None, llvm_vectorize: bool = False
) -> Path:
    """Compile LLVM IR for the host into a shared library (for ctypes / JIT)."""
    out_dir = out_dir or Path(tempfile.mkdtemp(prefix="algeco-"))
    lib = out_dir / "kernel.so"
    vec_flags = [] if llvm_vectorize else NO_LLVM_VECTORIZE
    run_tool(
        "clang",
        ["-x", "ir", "-", "-O2", *vec_flags, "-march=native", "-shared", "-fPIC"]
        + ["-o", str(lib)],
        llvm_ir,
    )
    return lib
