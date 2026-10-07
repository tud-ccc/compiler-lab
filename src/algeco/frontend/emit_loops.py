"""The `LoopEmitter`: expression tree -> loops on buffers (`scf` + `memref` + `arith`).

Every node becomes one loop nest that writes its result into a buffer:

- the root node writes into the caller's output buffer `%out`
- every other node gets a fresh `memref.alloc`, freed at the end of the function

Broadcasts too: the bias row `%b1` is copied into a full B x m buffer, which
the add then reads like any other operand, and so is the zero of a relu,
`max(x, broadcast(0))`. This is the naive translation; removing the copy is a
fusion (with the operation that reads it).

For `relu(x @ w1 + b1)` this emits::

    func.func @f(%x: memref<8x64xf32>, %w1: memref<64x256xf32>,
                 %b1: memref<256xf32>, %out: memref<8x256xf32>) {
      %zero = memref.get_global @cst_zero : memref<f32>
      %t0 = memref.alloc() : memref<8x256xf32>
      // matmul: init, then compute
      scf.for %i ... { scf.for %j ... { %t0[i, j] = 0 } }
      scf.for %i ... { scf.for %j ... { scf.for %k ... {
        %t0[i, j] = %t0[i, j] + %x[i, k] * %w1[k, j] } } }
      // broadcast b1 to every row
      %t1 = memref.alloc() : memref<8x256xf32>
      scf.for %i ... { scf.for %j ... { %t1[i, j] = %b1[j] } }
      // add
      %t2 = memref.alloc() : memref<8x256xf32>
      scf.for %i ... { scf.for %j ... { %t2[i, j] = %t0[i, j] + %t1[i, j] } }
      // broadcast the zero (a 0-d buffer) to every element
      %t3 = memref.alloc() : memref<8x256xf32>
      scf.for %i ... { scf.for %j ... { %t3[i, j] = %zero[] } }
      // maximum
      scf.for %i ... { scf.for %j ... { %out[i, j] = max(%t2[i, j], %t3[i, j]) } }
      memref.dealloc %t3
      memref.dealloc %t2
      memref.dealloc %t1
      memref.dealloc %t0
    }

Constants (`Const`, e.g. captured weights) become read-only globals that the
function reads with `memref.get_global`::

    "memref.global"() <{sym_name = "cst_w1", type = memref<64x256xf32>,
                        initial_value = dense<...>, constant, ...}>
    func.func @f(%x: memref<8x64xf32>, %out: memref<8x256xf32>) {
      %w1 = memref.get_global @cst_w1 : memref<64x256xf32>
      ...

All loop nests are perfectly nested and the matmul accumulates in memory, so
loop interchange and tiling are pure rearrangements of `scf.for` loops.

Buffers that are written (`%out` and the `memref.alloc`s) never overlap any
other buffer: allocations are fresh, and `%out` is fresh by the calling
convention of `algeco.jit`. Passes may rely on this.

Floating-point arithmetic is strict (no `fastmath` flags): every sum is computed
in the order of its loop, as in C. Interchange and tiling keep that order for
each element, so they must produce bit-identical results. The permission to
reorder reductions belongs to the higher-level `algeco` dialect, and is added by
its lowering.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from xdsl.builder import Builder
from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.builtin import (
    DenseIntOrFPElementsAttr,
    FloatAttr,
    IndexType,
    IntegerAttr,
    MemRefType,
    ModuleOp,
    StringAttr,
    TensorType,
    UnitAttr,
    f32,
)
from xdsl.ir import Block, Operation, Region, SSAValue
from xdsl.rewriter import InsertPoint

from algeco.frontend.expr import (
    Add,
    Broadcast,
    Const,
    Elementwise,
    Expr,
    MatMul,
    Maximum,
    Var,
)

INDEX_NAMES = ("i", "j", "k", "l", "m", "n")


def memref_type(shape: Sequence[int]) -> MemRefType:
    return MemRefType(f32, shape)


class LoopEmitter:
    """Emits a `func.func` computing an expression tree with explicit loops."""

    def __init__(self):
        self.builder: Builder
        self.entry: Block
        self.last_at_top: Operation | None = None
        self.constants: dict[tuple[object, object], SSAValue] = {}
        self.globals: list[memref.GlobalOp] = []
        self.global_values: dict[tuple[object, bytes], SSAValue] = {}
        self.symbols: set[str] = set()
        self.buffers: dict[Expr, SSAValue] = {}  # node -> buffer holding its value
        self.allocs: list[SSAValue] = []

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def emit_module(self, name: str, params: Sequence[Var], result: Expr) -> ModuleOp:
        """Emit `func.func @name(params..., %out)` computing `result` into `%out`."""
        arg_types = [memref_type(p.shape) for p in params] + [memref_type(result.shape)]
        self.symbols.add(name)
        self.entry = Block(arg_types=arg_types)
        self.builder = Builder(InsertPoint.at_end(self.entry))

        for param, arg in zip(params, self.entry.args[:-1], strict=True):
            arg.name_hint = param.name
            self.buffers[param] = arg
        out = self.entry.args[-1]
        out.name_hint = "out"

        self.emit(result, dest=out)

        for buffer in reversed(self.allocs):
            self.builder.insert(memref.DeallocOp.get(buffer))
        self.builder.insert(func.ReturnOp())

        function = func.FuncOp(name, (arg_types, []), Region(self.entry))
        module = ModuleOp([*self.globals, function])
        module.verify()
        return module

    # ------------------------------------------------------------------
    # Visitor: one method per node type
    # ------------------------------------------------------------------

    def emit(self, node: Expr, dest: SSAValue | None = None) -> SSAValue:
        """Emit `node` (once) and return the buffer that holds its value.

        `dest` is the buffer to write into; if None, a fresh buffer is allocated.
        """
        if isinstance(node, Const) and node not in self.buffers:
            self.buffers[node] = self.global_constant(node)
        if node in self.buffers:
            if dest is not None:  # e.g. the function just returns one of its inputs
                self.builder.insert(memref.CopyOp(self.buffers[node], dest))
                return dest
            return self.buffers[node]

        visit = getattr(self, f"visit_{type(node).__name__}")
        # Operands first, so their loop nests come before ours.
        operands = [self.emit(op) for op in node.operands]
        if dest is None:
            dest = self.alloc(node.shape)
        visit(node, operands, dest)
        self.buffers[node] = dest
        return dest

    def visit_MatMul(self, node: MatMul, operands: list[SSAValue], dest: SSAValue):
        a, b = operands
        m, n = node.shape
        k_size = node.lhs.shape[1]
        zero = self.float_constant(0.0)

        # dest[i, j] = 0
        with self.loop_nest((m, n)) as (i, j):
            self.builder.insert(memref.StoreOp.get(zero, dest, [i, j]))

        # dest[i, j] += a[i, k] * b[k, j]
        with self.loop_nest((m, n, k_size)) as (i, j, k):
            a_ik = self.builder.insert(memref.LoadOp.get(a, [i, k])).res
            b_kj = self.builder.insert(memref.LoadOp.get(b, [k, j])).res
            c_ij = self.builder.insert(memref.LoadOp.get(dest, [i, j])).res
            prod = self.builder.insert(arith.MulfOp(a_ik, b_kj)).result
            acc = self.builder.insert(arith.AddfOp(c_ij, prod)).result
            self.builder.insert(memref.StoreOp.get(acc, dest, [i, j]))

    def visit_Add(self, node: Add, operands: list[SSAValue], dest: SSAValue):
        self.elementwise(node, operands, dest, arith.AddfOp)

    def visit_Maximum(self, node: Maximum, operands: list[SSAValue], dest: SSAValue):
        self.elementwise(node, operands, dest, arith.MaximumfOp)

    def elementwise(
        self,
        node: Elementwise,
        operands: list[SSAValue],
        dest: SSAValue,
        op: type[arith.FloatingPointLikeBinaryOperation],
    ):
        """dest[ivs] = op(lhs[ivs], rhs[ivs])"""
        lhs, rhs = operands
        with self.loop_nest(node.shape) as ivs:
            x = self.builder.insert(memref.LoadOp.get(lhs, ivs)).res
            y = self.builder.insert(memref.LoadOp.get(rhs, ivs)).res
            r = self.builder.insert(op(x, y)).result
            self.builder.insert(memref.StoreOp.get(r, dest, ivs))

    def visit_Broadcast(
        self, node: Broadcast, operands: list[SSAValue], dest: SSAValue
    ):
        """dest[ivs] = src[ivs, with the repeated dimensions dropped or set to 0]."""
        (src,) = operands
        with self.loop_nest(node.shape) as ivs:
            x = self.load_broadcast(src, node.operand.shape, ivs)
            self.builder.insert(memref.StoreOp.get(x, dest, ivs))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @contextmanager
    def loop_nest(self, shape: Sequence[int]) -> Iterator[list[SSAValue]]:
        """Emit perfectly nested `scf.for` loops over `shape`.

        Yields the induction variables. Inside the `with` block, `self.builder`
        inserts into the innermost loop body.
        """
        outer_builder = self.builder
        bodies: list[Block] = []
        ivs: list[SSAValue] = []
        for d, dim in enumerate(shape):
            name = INDEX_NAMES[d] if d < len(INDEX_NAMES) else f"i{d}"
            body = Block(arg_types=[IndexType()])
            body.args[0].name_hint = name
            self.builder.insert(
                scf.ForOp(
                    self.index_constant(0),
                    self.index_constant(dim),
                    self.index_constant(1),
                    [],
                    body,
                )
            )
            self.builder = Builder(InsertPoint.at_end(body))
            bodies.append(body)
            ivs.append(body.args[0])
        try:
            yield ivs
        finally:
            for body in bodies:
                Builder(InsertPoint.at_end(body)).insert(scf.YieldOp())
            self.builder = outer_builder

    def load_broadcast(
        self, buffer: SSAValue, shape: Sequence[int], ivs: Sequence[SSAValue]
    ) -> SSAValue:
        """Load the element of a broadcast operand at output position `ivs`.

        NumPy broadcasting: shapes are aligned from the right; missing dimensions
        and dimensions of size 1 are repeated.
        """
        offset = len(ivs) - len(shape)
        indices = [
            self.index_constant(0) if size == 1 else ivs[offset + d]
            for d, size in enumerate(shape)
        ]
        return self.builder.insert(memref.LoadOp.get(buffer, indices)).res

    def alloc(self, shape: Sequence[int]) -> SSAValue:
        buffer = self.builder.insert(memref.AllocOp.get(f32, shape=shape)).memref
        buffer.name_hint = f"t{len(self.allocs)}"
        self.allocs.append(buffer)
        return buffer

    def index_constant(self, value: int) -> SSAValue:
        return self.constant(IntegerAttr(value, IndexType()), f"c{value}")

    def float_constant(self, value: float) -> SSAValue:
        name = "c0_f32" if value == 0.0 else "cst_f32"  # like MLIR's %c0_i32
        return self.constant(FloatAttr(value, f32), name)

    def constant(self, attr, name: str) -> SSAValue:
        """Constants go to the top of the function, so every loop nest can use them."""
        key = (type(attr), attr)
        if key not in self.constants:
            op = arith.ConstantOp(attr)
            op.result.name_hint = name
            self.insert_at_top(op)
            self.constants[key] = op.result
        return self.constants[key]

    def global_constant(self, node: Const) -> SSAValue:
        """A read-only global with the values of `node`; equal constants share one."""
        key = (node.shape, node.value.tobytes())
        if key not in self.global_values:
            # Prefixed, so that a constant called e.g. `malloc` or `free` cannot
            # clash with the C library functions the lowered code calls.
            symbol = self.unique_symbol(f"cst_{node.name}" if node.name else "cst")
            data = DenseIntOrFPElementsAttr.from_list(
                TensorType(f32, node.shape), node.value.ravel().tolist()
            )
            self.globals.append(
                memref.GlobalOp.get(
                    StringAttr(symbol),
                    memref_type(node.shape),
                    data,
                    constant=UnitAttr(),
                    alignment=64,
                )
            )
            op = memref.GetGlobalOp(symbol, memref_type(node.shape))
            op.memref.name_hint = node.name or symbol
            self.insert_at_top(op)
            self.global_values[key] = op.memref
        return self.global_values[key]

    def unique_symbol(self, base: str) -> str:
        symbol, n = base, 0
        while symbol in self.symbols:
            n += 1
            symbol = f"{base}_{n}"
        self.symbols.add(symbol)
        return symbol

    def insert_at_top(self, op: Operation) -> None:
        """Insert after the previous constant, at the start of the function."""
        if self.last_at_top is None:
            ip = InsertPoint.at_start(self.entry)
        else:
            ip = InsertPoint.after(self.last_at_top)
        Builder(ip).insert(op)
        self.last_at_top = op
