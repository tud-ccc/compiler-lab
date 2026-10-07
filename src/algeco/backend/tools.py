"""Locating and invoking the external toolchain (provided by pixi)."""

import os
import shutil
import subprocess
from pathlib import Path


class ToolError(RuntimeError):
    pass


def find_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise ToolError(
            f"'{name}' not found on PATH. Did you activate the pixi environment "
            "(`pixi shell` or `direnv allow`)?"
        )
    return path


def run_tool(name: str, args: list[str], stdin: str | bytes) -> bytes:
    data = stdin.encode() if isinstance(stdin, str) else stdin
    proc = subprocess.run([find_tool(name), *args], input=data, capture_output=True)
    if proc.returncode != 0:
        raise ToolError(f"{name} failed:\n{proc.stderr.decode()}")
    return proc.stdout


def mlir_lib_dir() -> Path:
    return Path(
        os.environ.get("ALGECO_MLIR_LIB_DIR")
        or Path(find_tool("mlir-opt")).parent.parent / "lib"
    )


def mlir_runtime_lib(name: str) -> Path:
    """Path to an MLIR runtime library, e.g. `mlir_c_runner_utils`.

    conda-forge only ships versioned sonames (libfoo.so.23.1), so we glob.
    """
    lib_dir = mlir_lib_dir()
    for pattern in (f"lib{name}.so*", f"lib{name}.dylib", f"lib{name}.*.dylib"):
        if hits := sorted(lib_dir.glob(pattern)):
            return hits[0]
    raise ToolError(f"MLIR runtime library '{name}' not found in {lib_dir}")
