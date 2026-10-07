"""`algeco.jit`: compile a Python function on arrays with algeco.

Usage::

    @algeco.jit
    def mlp(x, w1, b1, w2, b2):
        return algeco.relu(x @ w1 + b1) @ w2 + b2

    y = mlp(x, w1, b1, w2, b2)      # NumPy arrays in, NumPy array out
    print(mlp.mlir(x, w1, b1, w2, b2))  # the IR after your passes

    @algeco.jit(passes="algeco-interchange")   # run your own passes
    def ...

    @algeco.jit(dump="dump/")   # write the IR of every stage to dump/<name>_<shapes>/
    def ...                     # (see `algeco.dump`), with a compile.log

Arrays the function captures (from an enclosing function or a global) are
compiled into the code as constants, while parameters stay runtime inputs.
This lets you compile in stages, e.g. specialize the MLP to trained weights::

    def compile_mlp(w1, b1, w2, b2):
        @algeco.jit
        def infer(x):                   # only x is an input
            return np.maximum(x @ w1 + b1, 0) @ w2 + b2
        return infer

    infer = compile_mlp(w1, b1, w2, b2)
    y = infer(x)

Captured arrays are read when compiling for a new input shape. Changing them
afterwards does not affect code that is already compiled.

On the first call with a given set of input shapes, algeco

1. calls the function on `Var` placeholders, which builds an expression tree,
2. emits loops for it (`LoopEmitter`),
3. runs the pass pipeline,
4. lowers to LLVM IR, compiles a shared library, and loads it.

Later calls with the same shapes reuse the compiled code.

Calling convention: the compiled function receives one pointer per input and
one for the output, and **the output never overlaps an input** (it is always a
fresh array). Inputs may share memory (e.g. `f(x, x)`), but they are only read.
So a pass may assume that a buffer the function writes is not also visible
under another name, as C's `restrict` or Fortran arguments do.
"""

import inspect
import re
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from xdsl.dialects.builtin import ModuleOp

from algeco.backend import CompiledFunction, build_shared_library, compile_llvm_ir
from algeco.backend.runtime import as_buffers
from algeco.context import make_context
from algeco.dump import Dump
from algeco.frontend.emit_loops import LoopEmitter
from algeco.frontend.expr import Const, Expr, Var, walk
from algeco.pipeline import run_passes, to_llvm_ir, to_text

Shapes = tuple[tuple[int, ...], ...]


@dataclass
class Compiled:
    function: CompiledFunction
    out_shape: tuple[int, ...]
    mlir: str
    llvm_ir: str


class JitFunction:
    def __init__(
        self,
        fn: Callable[..., Expr],
        passes: str = "",
        llvm_vectorize: bool = False,
        dump: str | Path | None = None,
    ):
        self.fn = fn
        self.name = fn.__name__
        # Symbol of the compiled function. The prefix keeps it from clashing with
        # C library functions the generated code calls (e.g. `malloc`, `free`).
        self.symbol = "algeco_" + re.sub(r"\W", "_", self.name)
        self.param_names = list(inspect.signature(fn).parameters)
        self.passes = passes
        self.llvm_vectorize = llvm_vectorize
        self.dump = Path(dump) if dump is not None else None
        self.cache: dict[Shapes, Compiled] = {}

    def expr(self, *shapes: tuple[int, ...]) -> tuple[list[Var], Expr]:
        """Call the function on placeholders; returns the parameters and the tree."""
        if len(shapes) != len(self.param_names):
            raise TypeError(f"{self.name}() takes {len(self.param_names)} arguments")
        params = [
            Var(name, tuple(shape))
            for name, shape in zip(self.param_names, shapes, strict=True)
        ]
        result = self.fn(*params)
        if not isinstance(result, Expr):
            raise TypeError(f"{self.name}() must return an algeco expression")
        self.name_constants(result)
        return params, result

    def name_constants(self, result: Expr) -> None:
        """Name constants after the captured variables that hold them (for the IR)."""
        code = self.fn.__code__
        names: dict[int, str] = {}
        for name, cell in zip(code.co_freevars, self.fn.__closure__ or (), strict=True):
            try:
                names.setdefault(id(cell.cell_contents), name)
            except ValueError:  # cell not yet assigned
                pass
        for name in code.co_names:
            if name in self.fn.__globals__:
                names.setdefault(id(self.fn.__globals__[name]), name)
        for node in walk(result):
            if isinstance(node, Const) and node.name is None:
                node.name = names.get(id(node.source))

    def emit(self, *arrays: np.ndarray) -> ModuleOp:
        """The IR emitted by the frontend, before any passes."""
        params, result = self.expr(*(np.shape(a) for a in arrays))
        return LoopEmitter().emit_module(self.symbol, params, result)

    def mlir(self, *arrays: np.ndarray) -> str:
        """The IR after the pass pipeline, as handed to the backend."""
        return self.compile(*arrays).mlir

    def compile(self, *arrays: np.ndarray) -> Compiled:
        """
        Run the full compilation pipeline on the function, keyed
        by array shapes in the cache.
        """
        shapes = tuple(tuple(np.shape(a)) for a in arrays)
        if shapes not in self.cache:
            dump = self.open_dump(shapes)
            try:
                self.cache[shapes] = self.compile_shapes(shapes, dump)
            except Exception as e:
                if dump:
                    dump.log(f"FAILED: {type(e).__name__}: {e}")
                raise
            finally:
                if dump:
                    dump.close()
        return self.cache[shapes]

    def compile_shapes(self, shapes: Shapes, dump: Dump | None) -> Compiled:
        ctx = make_context()
        # 1. Turn the Expr DAG into an xdsl module
        params, result = self.expr(*shapes)
        if dump:
            dump.stage(
                "expr",
                f"{self.name}({', '.join(map(str, params))}) =\n  {result}\n",
                ".txt",
            )
        module = LoopEmitter().emit_module(self.symbol, params, result)
        if dump:
            dump.stage("emitted", to_text(module))
        # 2. Middle-end optimization
        run_passes(ctx, module, self.passes, dump)
        module.verify()
        # 3. Backend code generation.
        llvm_ir = to_llvm_ir(module, dump)
        if dump:
            out_dir = dump.directory
        else:
            out_dir = Path(tempfile.mkdtemp(prefix=f"{self.symbol}-"))
        library = build_shared_library(llvm_ir, out_dir, self.llvm_vectorize)
        if dump:
            dump.done(library.name)
            asm = compile_llvm_ir(llvm_ir, "asm", llvm_vectorize=self.llvm_vectorize)
            dump.stage("asm", asm, ".s")

        return Compiled(
            function=CompiledFunction(library, self.symbol, len(shapes) + 1),
            out_shape=result.shape,
            mlir=to_text(module),
            llvm_ir=llvm_ir,
        )

    def open_dump(self, shapes: Shapes) -> Dump | None:
        """A dump directory for this function and these shapes, if requested."""
        if self.dump is None:
            return None
        name = "_".join(
            [self.name, *("x".join(map(str, s)) or "scalar" for s in shapes)]
        )
        dump = Dump(self.dump / name)
        args = ", ".join(
            f"{p}: {s}" for p, s in zip(self.param_names, shapes, strict=True)
        )
        dump.log(f"{self.name}({args})")
        dump.log(f"passes: {self.passes or '(none)'}")
        dump.log(f"llvm_vectorize: {self.llvm_vectorize}")
        dump.log("")
        dump.log(f"{'total':>10}  {'stage':>10}  output")
        return dump

    def __call__(self, *arrays: np.ndarray) -> np.ndarray:
        compiled = self.compile(*arrays)
        inputs = as_buffers(arrays)
        out = np.empty(compiled.out_shape, dtype=np.float32)  # never aliases inputs
        compiled.function(*inputs, out)
        return out


def jit(
    fn=None,
    *,
    passes: str = "",
    llvm_vectorize: bool = False,
    dump: str | Path | None = None,
):
    """Decorator; use as `@jit` or `@jit(passes="...", dump="dump/")`."""
    if fn is None:
        return lambda f: JitFunction(f, passes, llvm_vectorize, dump)
    return JitFunction(fn, passes, llvm_vectorize, dump)
