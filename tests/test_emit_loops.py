"""Invariants of the IR emitted by the LoopEmitter.

These hold for every program and are what the Task 1 passes may rely on.
The exact shape of the emitted IR is tested with FileCheck in
tests/filecheck/frontend/.
"""

import numpy as np
import pytest
from xdsl.dialects import arith, func, memref, scf
from xdsl.dialects.builtin import IntegerAttr, ModuleOp

from algeco.frontend import Const, Expr, Var, broadcast_to, relu
from algeco.frontend.emit_loops import LoopEmitter

W, B, W3 = np.ones((3, 4), np.float32), np.ones(4, np.float32), np.eye(3)


def program(name, shapes, build):
    params = [Var(f"p{n}", shape) for n, shape in enumerate(shapes)]
    return pytest.param(params, build(*params), id=name)


PROGRAMS = [
    program("matmul", [(2, 3), (3, 4)], lambda a, b: a @ b),
    program("add", [(2, 3), (2, 3)], lambda a, b: a + b),
    program("add_bias", [(2, 3), (3,)], lambda a, b: a + b),
    program("add_outer", [(1, 3), (2, 1)], lambda a, b: a + b),
    program("relu", [(5,)], lambda a: relu(a)),
    program("relu_relu", [(2, 2)], lambda a: relu(relu(a))),
    program("matmul_of_sum", [(2, 3), (2, 3), (3, 4)], lambda a, b, w: (a + b) @ w),
    program("rank3_broadcast", [(2, 1, 4), (3, 1)], lambda a, b: relu(a + b)),
    program("rank7", [(2, 1, 2, 1, 2, 1, 2), (2,)], lambda a, b: a + b),
    program(
        "mlp",
        [(4, 8), (8, 6), (6,), (6, 3), (3,)],
        lambda x, w1, b1, w2, b2: relu(x @ w1 + b1) @ w2 + b2,
    ),
    program("shared", [(2, 2), (2, 2)], lambda a, b: (lambda t: t + t)(a @ b)),
    program("identity", [(2, 3)], lambda a: a),
    program("const_weights", [(2, 3)], lambda x: relu(x @ W + B)),
    program("const_scalar", [(2, 3)], lambda x: x + 1.0),
    program("const_shared", [(3, 3)], lambda x: x @ W3 @ W3),
    program("const_root", [(2, 3)], lambda x: Const(W)),
    program("relu_broadcast", [(3,)], lambda a: relu(broadcast_to(a, (2, 3)))),
    program(
        "broadcast_matmul", [(3,), (3, 4)], lambda b, w: broadcast_to(b, (2, 3)) @ w
    ),
    program("broadcast_root", [(3,)], lambda b: broadcast_to(b, (2, 3))),
]


def emit(params: list[Var], result: Expr) -> func.FuncOp:
    module = LoopEmitter().emit_module("f", params, result)
    module.verify()
    (function,) = [op for op in module.ops if isinstance(op, func.FuncOp)]
    return function


def top_level_ops(function: func.FuncOp):
    return list(function.body.block.ops)


def constant_value(value) -> int:
    op = value.owner
    assert isinstance(op, arith.ConstantOp), f"loop bound is not a constant: {op}"
    assert isinstance(op.value, IntegerAttr)
    return op.value.value.data


def stores_into(op, buffer) -> bool:
    return any(
        (isinstance(o, memref.StoreOp) and o.memref == buffer)
        or (isinstance(o, memref.CopyOp) and o.destination == buffer)
        for o in op.walk()
    )


@pytest.mark.parametrize("params, result", PROGRAMS)
def test_loop_nests_are_perfect(params, result):
    """Every loop body contains either exactly one inner loop, or no loop at all."""
    for loop in emit(params, result).walk():
        if not isinstance(loop, scf.ForOp):
            continue
        body = list(loop.body.block.ops)
        inner_loops = [o for o in body if isinstance(o, scf.ForOp)]
        if inner_loops:
            assert len(body) == 2 and isinstance(body[1], scf.YieldOp), loop


@pytest.mark.parametrize("params, result", PROGRAMS)
def test_loops_have_constant_bounds_and_unit_steps(params, result):
    for loop in emit(params, result).walk():
        if isinstance(loop, scf.ForOp):
            assert constant_value(loop.lb) == 0
            assert constant_value(loop.step) == 1
            assert constant_value(loop.ub) > 0
            assert (
                loop.iter_args == ()
            )  # no loop-carried values: all state is in memory


@pytest.mark.parametrize("params, result", PROGRAMS)
def test_constants_are_unique_and_at_the_top(params, result):
    function = emit(params, result)
    ops = top_level_ops(function)
    n_constants = sum(isinstance(o, arith.ConstantOp) for o in function.walk())
    # all constants are at the top of the function body, none inside loops
    top = [o for o in ops if isinstance(o, arith.ConstantOp | memref.GetGlobalOp)]
    assert sum(isinstance(o, arith.ConstantOp) for o in top) == n_constants
    values = [(o.value, o.result.type) for o in top if isinstance(o, arith.ConstantOp)]
    assert len(values) == len(set(values))


@pytest.mark.parametrize("params, result", PROGRAMS)
def test_every_alloc_is_freed_once_at_the_end(params, result):
    function = emit(params, result)
    ops = top_level_ops(function)
    allocs = [o.memref for o in ops if isinstance(o, memref.AllocOp)]
    assert isinstance(ops[-1], func.ReturnOp)
    tail = ops[len(ops) - 1 - len(allocs) : -1]
    assert all(isinstance(o, memref.DeallocOp) for o in tail)
    assert [o.memref for o in tail] == allocs[::-1]
    assert sum(isinstance(o, memref.DeallocOp) for o in function.walk()) == len(allocs)


@pytest.mark.parametrize("params, result", PROGRAMS)
def test_inputs_and_constants_are_never_written(params, result):
    function = emit(params, result)
    inputs = function.body.block.args[:-1]
    constants = [o.memref for o in function.walk() if isinstance(o, memref.GetGlobalOp)]
    for buffer in (*inputs, *constants):
        assert not stores_into(function, buffer), f"%{buffer.name_hint} is written"


@pytest.mark.parametrize("params, result", PROGRAMS)
def test_constants_are_read_at_the_top(params, result):
    function = emit(params, result)
    ops = top_level_ops(function)
    at_top = [o for o in ops if isinstance(o, arith.ConstantOp | memref.GetGlobalOp)]
    assert ops[: len(at_top)] == at_top


@pytest.mark.parametrize("params, result", PROGRAMS)
def test_only_the_last_nests_write_the_output(params, result):
    """The root writes into %out, at the end; no other buffer is written after that."""
    function = emit(params, result)
    out = function.body.block.args[-1]
    nests = [
        o
        for o in top_level_ops(function)
        if not isinstance(
            o, arith.ConstantOp | memref.GetGlobalOp | memref.AllocOp | memref.DeallocOp
        )
        and not isinstance(o, func.ReturnOp)
    ]
    writes_out = [stores_into(n, out) for n in nests]
    assert writes_out[-1]
    first = writes_out.index(True)
    assert all(writes_out[first:]), "an unrelated nest comes after the output nests"


@pytest.mark.parametrize(
    "params, result, expected",
    [
        (p.values[0], p.values[1], n)
        for p, n in zip(
            PROGRAMS,
            # allocations: one per intermediate node (not inputs, not the root),
            # broadcasts included (a relu broadcasts its zero)
            [0, 0, 1, 2, 1, 3, 1, 4, 1, 7, 1, 0, 4, 1, 1, 0, 2, 1, 0],
            strict=True,
        )
    ],
    ids=[p.id for p in PROGRAMS],
)
def test_one_buffer_per_intermediate(params, result, expected):
    function = emit(params, result)
    assert sum(isinstance(o, memref.AllocOp) for o in function.walk()) == expected


def test_nests_per_operation():
    a, b, w = Var("a", (2, 3)), Var("b", (2, 3)), Var("w", (3, 4))
    function = emit([a, b, w], relu((a + b) @ w))
    nests = [o for o in top_level_ops(function) if isinstance(o, scf.ForOp)]
    # add: 1, matmul: 2 (init + compute), relu: 2 (broadcast zero + maximum)
    assert len(nests) == 5


def test_emitter_does_not_modify_module_type():
    x = Var("x", (2,))
    module = LoopEmitter().emit_module("f", [x], relu(x))
    assert isinstance(module, ModuleOp)
