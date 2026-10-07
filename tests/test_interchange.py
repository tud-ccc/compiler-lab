"""Tests for the loop interchange pass (Task 1).

Three layers, like the pass itself:
- the decisions (`is_legal`, `choose_order`) on hand-written models, no IR
- the pass on IR: what it changes, and what it must leave alone
- end to end: compiled results stay bit-identical
"""

import itertools

import numpy as np
import pytest
from xdsl.dialects import scf

import algeco
from algeco.context import make_context
from algeco.frontend import Var, relu
from algeco.frontend.emit_loops import LoopEmitter
from algeco.pipeline import run_passes, to_text
from algeco.transforms.interchange import (
    Access,
    choose_order,
    extract_accesses,
    is_legal,
    perfect_nest,
)

# The matmul nest i, j, k: out[i, j] += x[i, k] * w[k, j]
MATMUL = [
    Access("x", (0, 2), False),
    Access("w", (2, 1), False),
    Access("out", (0, 1), False),
    Access("out", (0, 1), True),
]

# ---------------------------------------------------------------------------
# Decisions, on the model only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_every_matmul_order_is_legal(order):
    # only k is a reduction loop, and a single loop always keeps its order
    assert is_legal(MATMUL, order)


def test_reduction_loops_must_keep_their_relative_order():
    # out[i] += a[i, j, k]: j and k are both reduction loops
    accesses = [
        Access("a", (0, 1, 2), False),
        Access("out", (0,), False),
        Access("out", (0,), True),
    ]
    assert is_legal(accesses, (1, 0, 2))  # j, i, k: j still before k
    assert is_legal(accesses, (1, 2, 0))
    assert not is_legal(accesses, (0, 2, 1))  # k before j: different sum order
    assert not is_legal(accesses, (2, 1, 0))


def test_written_buffer_accessed_differently_is_illegal():
    # a[i, j] = a[j, i]
    transpose = [Access("a", (1, 0), False), Access("a", (0, 1), True)]
    assert not is_legal(transpose, (1, 0))


def test_written_buffer_with_constant_index_is_illegal():
    accesses = [Access("a", (0, None), False), Access("a", (0, None), True)]
    assert not is_legal(accesses, (1, 0))


def test_reads_alone_never_restrict():
    accesses = [Access("a", (1, 0), False), Access("out", (0, 1), True)]
    assert is_legal(accesses, (1, 0))


def test_cost_model_picks_ikj_for_matmul():
    assert choose_order(MATMUL, [64, 256, 64]) == (0, 2, 1)
    assert choose_order(MATMUL, [64, 10, 256]) == (0, 2, 1)  # second MLP layer


def test_cost_model_keeps_a_good_order():
    elementwise = [Access("a", (0, 1), False), Access("out", (0, 1), True)]
    assert choose_order(elementwise, [64, 256]) == (0, 1)
    ikj = [Access("x", (0, 1), False), Access("w", (1, 2), False),
           Access("out", (0, 2), True)]  # fmt: skip
    assert choose_order(ikj, [64, 64, 256]) == (0, 1, 2)


def test_cost_model_fixes_a_transposed_traversal():
    # out[i, j] = a[i, j], but with j outside: walks both arrays column by column
    accesses = [Access("a", (1, 0), False), Access("out", (1, 0), True)]
    assert choose_order(accesses, [64, 64]) == (1, 0)


# ---------------------------------------------------------------------------
# The pass on IR
# ---------------------------------------------------------------------------


def emit(build, *shapes):
    params = [Var(f"p{n}", shape) for n, shape in enumerate(shapes)]
    return LoopEmitter().emit_module("f", params, build(*params))


def interchange(module, passes="algeco-interchange"):
    run_passes(make_context(), module, passes)
    module.verify()
    return module


def nests(module):
    """The loop variable names of every outermost nest, e.g. ['i', 'k', 'j']."""
    result = []
    for op in module.walk():
        if isinstance(op, scf.ForOp) and not isinstance(op.parent_op(), scf.ForOp):
            nest = perfect_nest(op) or [op]
            result.append([loop.body.block.args[0].name_hint for loop in nest])
    return result


def test_matmul_becomes_ikj_and_elementwise_stays():
    module = interchange(emit(lambda x, w, b: relu(x @ w + b), (8, 4), (4, 16), (16,)))
    # zero-fill, matmul, broadcast b, add, broadcast 0, maximum
    assert nests(module) == [
        ["i", "j"], ["i", "k", "j"], ["i", "j"], ["i", "j"], ["i", "j"], ["i", "j"]
    ]  # fmt: skip


def test_pass_is_idempotent():
    module = interchange(emit(lambda x, w: x @ w, (8, 4), (4, 16)))
    before = to_text(module)
    assert to_text(interchange(module)) == before


def test_forced_order():
    module = interchange(
        emit(lambda x, w: x @ w, (8, 4), (4, 16)), "algeco-interchange{order=2,0,1}"
    )
    assert nests(module)[1] == ["k", "i", "j"]


def test_forced_order_of_wrong_depth_is_ignored():
    module = interchange(
        emit(lambda x, w: x @ w, (8, 4), (4, 16)), "algeco-interchange{order=1,0}"
    )
    # the 2-deep zero-fill nest is permuted, the 3-deep matmul nest is not
    assert nests(module) == [["j", "i"], ["i", "j", "k"]]


LEFT_ALONE = {
    "imperfect nest": """
      scf.for %i = %c0 to %c4 step %c1 {
        memref.store %zero, %out[%i, %c0] : memref<4x4xf32>
        scf.for %j = %c0 to %c4 step %c1 {
          %v = memref.load %a[%j, %i] : memref<4x4xf32>
          memref.store %v, %out[%i, %j] : memref<4x4xf32>
        }
      }""",
    "computed index": """
      scf.for %i = %c0 to %c3 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %i1 = arith.addi %i, %c1 : index
          %v = memref.load %a[%j, %i1] : memref<4x4xf32>
          memref.store %v, %out[%i, %j] : memref<4x4xf32>
        }
      }""",
    "in-place transpose": """
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %v = memref.load %out[%j, %i] : memref<4x4xf32>
          memref.store %v, %out[%i, %j] : memref<4x4xf32>
        }
      }""",
    "non-constant bound": """
      %n = memref.dim %a, %c0 : memref<4x4xf32>
      scf.for %i = %c0 to %n step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %v = memref.load %a[%j, %i] : memref<4x4xf32>
          memref.store %v, %out[%i, %j] : memref<4x4xf32>
        }
      }""",
    "loop-carried value": """
      scf.for %i = %c0 to %c4 step %c1 {
        %s = scf.for %j = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> (f32) {
          %v = memref.load %a[%j, %i] : memref<4x4xf32>
          %acc2 = arith.addf %acc, %v : f32
          scf.yield %acc2 : f32
        }
        memref.store %s, %out[%i, %c0] : memref<4x4xf32>
      }""",
}


@pytest.mark.parametrize("body", LEFT_ALONE.values(), ids=LEFT_ALONE.keys())
def test_leaves_unsupported_nests_alone(body):
    from xdsl.parser import Parser

    src = f"""
    func.func @f(%a: memref<4x4xf32>, %out: memref<4x4xf32>) {{
      %zero = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      {body}
      func.return
    }}"""
    module = Parser(make_context(), src).parse_module()
    before = to_text(module)
    assert to_text(interchange(module)) == before


def test_extract_accesses_of_matmul():
    module = emit(lambda x, w: x @ w, (8, 4), (4, 16))
    loops = [op for op in module.walk() if isinstance(op, scf.ForOp)]
    nest = perfect_nest(loops[2])  # after the 2-deep zero-fill nest
    accesses = extract_accesses(nest)
    assert [(str(a.buffer.name_hint), a.dims, a.is_write) for a in accesses] == [
        ("p0", (0, 2), False),
        ("p1", (2, 1), False),
        ("out", (0, 1), False),
        ("out", (0, 1), True),
    ]


# ---------------------------------------------------------------------------
# End to end: results stay bit-identical
# ---------------------------------------------------------------------------


def mlp(x, w1, b1, w2, b2):
    return algeco.relu(x @ w1 + b1) @ w2 + b2


@pytest.mark.parametrize(
    "passes",
    ["algeco-interchange"]
    + [
        "algeco-interchange{order=" + ",".join(map(str, p)) + "}"
        for p in itertools.permutations(range(3))
    ],
)
def test_mlp_results_are_bit_identical(passes):
    rng = np.random.default_rng(0)
    args = [
        rng.standard_normal(s, dtype=np.float32)
        for s in [(16, 32), (32, 24), (24,), (24, 10), (10,)]
    ]
    expected = algeco.jit(mlp)(*args)
    np.testing.assert_array_equal(algeco.jit(mlp, passes=passes)(*args), expected)
