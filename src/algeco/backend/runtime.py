"""Loading compiled kernels and calling them on NumPy arrays."""

import ctypes
from collections.abc import Sequence
from pathlib import Path

import numpy as np


class CompiledFunction:
    """A function in a shared library, taking one pointer per (float32) buffer.

    With the bare-pointer calling convention (see `backend/mlir.py`), a function
    `func.func @f(%a: memref<..>, %b: memref<..>)` becomes `void f(float *a, float *b)`.
    """

    def __init__(self, library: Path, name: str, num_args: int):
        self.library = ctypes.CDLL(
            str(library)
        )  # keep a reference: unloading frees the code
        self.function = getattr(self.library, name)
        self.function.argtypes = [ctypes.c_void_p] * num_args
        self.function.restype = None

    def __call__(self, *arrays: np.ndarray) -> None:
        for a in arrays:
            if a.dtype != np.float32 or not a.flags.c_contiguous:
                raise TypeError("arguments must be C-contiguous float32 arrays")
        self.function(*(a.ctypes.data for a in arrays))


def as_buffers(arrays: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.ascontiguousarray(a, dtype=np.float32) for a in arrays]
