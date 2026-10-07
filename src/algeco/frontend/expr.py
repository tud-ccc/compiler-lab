"""Expression trees: the input language of algeco.

Python operators on `Expr` objects don't compute anything. They build a tree
that describes the computation::

    x @ w1 + b1  ->  Add(MatMul(Var x, Var w1), Broadcast(Var b1))

`+` follows NumPy's broadcasting rules, like NumPy itself, but the tree makes
them explicit: `Add` only adds operands of equal shape, and the operator
inserts a `Broadcast` node for an operand that needs repeating (the bias row
`b1` is repeated for every row of the batch).

`relu` is not a node of its own: `relu(x)` is `maximum(x, 0)`, and the scalar
`0` is broadcast like the bias::

    relu(x)  ->  Maximum(Var x, Broadcast(Const zero))

NumPy arrays and Python numbers used as operands become constants (`Const`),
and the NumPy functions `np.add`, `np.matmul`, `np.maximum` and
`np.broadcast_to` also work on expressions. So weights captured from an
enclosing function are compiled into the code::

    def make_layer(w, b):
        @algeco.jit
        def layer(x):
            # Maximum(Add(MatMul(x, Const w), Broadcast(Const b)),
            #         Broadcast(Const 0))
            return np.maximum(x @ w + b, 0)
        return layer

The tree is then turned into IR by an emitter (see `emit_loops.py`).

All values are float32 arrays with shapes known at compile time.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np


class Expr:
    """Base class of all expression nodes."""

    shape: tuple[int, ...]

    @property
    def operands(self) -> tuple[Expr, ...]:
        return ()

    def __add__(self, other) -> Expr:
        return Add(self, other)

    def __radd__(self, other) -> Expr:
        return Add(other, self)

    def __matmul__(self, other) -> Expr:
        return MatMul(self, other)

    def __rmatmul__(self, other) -> Expr:
        return MatMul(other, self)

    # NumPy calls these when one of the arguments is an `Expr`, e.g. in
    # `np.maximum(e, 0)` or `array @ e`, so that they build a tree as well.

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__" and not kwargs:
            if ufunc is np.add:
                return Add(*inputs)
            if ufunc is np.matmul:
                return MatMul(*inputs)
            if ufunc is np.maximum:
                return Maximum(*inputs)
        name = ufunc.__name__ if method == "__call__" else f"{ufunc.__name__}.{method}"
        raise TypeError(f"algeco does not support np.{name}{unsupported_hint()}")

    def __array_function__(self, func, types, args, kwargs):
        if func is np.broadcast_to and not kwargs:
            return broadcast_to(*args)
        raise TypeError(
            f"algeco does not support np.{func.__name__}{unsupported_hint()}"
        )


def unsupported_hint() -> str:
    return (
        " (supported: +, @, relu, maximum, broadcast_to, np.add, np.matmul,"
        " np.maximum, np.broadcast_to)"
    )


def as_expr(value) -> Expr:
    """Expressions stay as they are; arrays and numbers become constants."""
    return value if isinstance(value, Expr) else Const(value)


def walk(root: Expr) -> Iterator[Expr]:
    """All nodes of the tree, each shared node once, operands before users."""
    seen: set[int] = set()

    def visit(node: Expr) -> Iterator[Expr]:
        if id(node) in seen:
            return
        seen.add(id(node))
        for operand in node.operands:
            yield from visit(operand)
        yield node

    yield from visit(root)


# eq=False: nodes are compared by identity. The same node object used twice
# (e.g. `t = x @ w; t + t`) is one shared subexpression, computed once.


@dataclass(eq=False)
class Var(Expr):
    """A function parameter."""

    name: str
    shape: tuple[int, ...]

    def __str__(self) -> str:
        return self.name


@dataclass(eq=False)
class Const(Expr):
    """A constant array, compiled into the code (e.g. trained weights).

    `source` is the object it was created from; `algeco.jit` uses it to name the
    constant after the variable that holds it. The values are converted to a
    float32 copy only if necessary, so they are read when the code is compiled.
    """

    source: object = field(repr=False)
    name: str | None = None
    value: np.ndarray = field(init=False, repr=False)
    shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        value = np.asarray(self.source, dtype=np.float32)  # no copy if possible
        self.value = value if value.flags.c_contiguous else value.copy(order="C")
        self.shape = self.value.shape

    def __str__(self) -> str:
        return self.name or f"const{list(self.shape)}"


@dataclass(eq=False)
class MatMul(Expr):
    """Matrix product of two 2-d operands: (M, K) @ (K, N) -> (M, N)."""

    lhs: Expr
    rhs: Expr
    shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        self.lhs, self.rhs = as_expr(self.lhs), as_expr(self.rhs)
        if len(self.lhs.shape) != 2 or len(self.rhs.shape) != 2:
            raise ValueError(
                f"matmul needs 2-d operands: {self.lhs.shape} @ {self.rhs.shape}"
            )
        (m, k), (k2, n) = self.lhs.shape, self.rhs.shape
        if k != k2:
            raise ValueError(
                f"matmul shape mismatch: {self.lhs.shape} @ {self.rhs.shape}"
            )
        self.shape = (m, n)

    @property
    def operands(self) -> tuple[Expr, ...]:
        return (self.lhs, self.rhs)

    def __str__(self) -> str:
        return f"({self.lhs} @ {self.rhs})"


@dataclass(eq=False)
class Broadcast(Expr):
    """Repeat an operand to a larger shape, following NumPy's rules.

    Shapes are aligned from the right; missing dimensions and dimensions of
    size 1 are repeated. E.g. a bias row (m,) becomes (B, m): every row is `b`.
    Create it with `broadcast_to`, which leaves operands of the right shape alone.
    """

    operand: Expr
    shape: tuple[int, ...]

    def __post_init__(self):
        self.operand = as_expr(self.operand)
        self.shape = tuple(self.shape)
        try:
            ok = np.broadcast_shapes(self.operand.shape, self.shape) == self.shape
        except ValueError:
            ok = False
        if not ok:
            raise ValueError(
                f"cannot broadcast shape {self.operand.shape} to {self.shape}"
            )

    @property
    def operands(self) -> tuple[Expr, ...]:
        return (self.operand,)

    def __str__(self) -> str:
        return f"broadcast({self.operand}, {self.shape})"


@dataclass(eq=False)
class Elementwise(Expr):
    """An elementwise operation on two operands of the same shape.

    Operands of different shapes are broadcast first, as in NumPy: the
    constructor wraps them in `Broadcast` nodes.
    """

    lhs: Expr
    rhs: Expr
    shape: tuple[int, ...] = field(init=False)

    def __post_init__(self):
        lhs, rhs = as_expr(self.lhs), as_expr(self.rhs)
        try:
            self.shape = np.broadcast_shapes(lhs.shape, rhs.shape)
        except ValueError:
            raise ValueError(
                f"{type(self).__name__.lower()}: shapes {lhs.shape} and"
                f" {rhs.shape} do not broadcast"
            ) from None
        self.lhs, self.rhs = (
            broadcast_to(lhs, self.shape),
            broadcast_to(rhs, self.shape),
        )

    @property
    def operands(self) -> tuple[Expr, ...]:
        return (self.lhs, self.rhs)


@dataclass(eq=False)
class Add(Elementwise):
    """Elementwise `lhs + rhs`."""

    def __str__(self) -> str:
        return f"({self.lhs} + {self.rhs})"


@dataclass(eq=False)
class Maximum(Elementwise):
    """Elementwise `max(lhs, rhs)`, like `np.maximum` (NaN if either is NaN)."""

    def __str__(self) -> str:
        return f"max({self.lhs}, {self.rhs})"


def matmul(lhs: Expr, rhs: Expr) -> Expr:
    return MatMul(lhs, rhs)


def broadcast_to(x, shape) -> Expr:
    """`x` repeated to `shape` (like `np.broadcast_to`); `x` itself if it fits."""
    x = as_expr(x)
    shape = tuple(shape)
    if x.shape == shape:
        return x
    if isinstance(x, Broadcast):  # broadcast(broadcast(y)) = broadcast(y)
        x = x.operand
    return Broadcast(x, shape)


def maximum(a, b) -> Expr:
    return Maximum(a, b)


def relu(x) -> Expr:
    """`max(x, 0)`: the zero is a constant, broadcast to the shape of `x`."""
    return Maximum(x, Const(np.float32(0), name="zero"))
