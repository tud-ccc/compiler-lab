"""Tests for loop fusion (Task 1).

- the pass on the frontend's loops: what gets fused, which buffers disappear
- hand-written nests it must not fuse, or must fuse carefully
- end to end: compiled results stay bit-identical, in any pipeline
"""

import numpy as np
import pytest
from loops import emit, outermost_loops, parse, transform
from xdsl.dialects import memref

import algeco
from algeco.frontend import relu
from algeco.pipeline import to_text
from algeco.transforms.fusion import can_fuse, loop_chain
from algeco.transforms.interchange import Access


def allocs(module):
    return [op for op in module.walk() if isinstance(op, memref.AllocOp)]


def stores(module):
    return [op for op in module.walk() if isinstance(op, memref.StoreOp)]


def chain(module, n=0):
    """The loop variable names down the n-th outermost nest."""
    nest = loop_chain(outermost_loops(module)[n])
    return [loop.body.block.args[0].name_hint for loop in nest]


# ---------------------------------------------------------------------------
# Legality, on the model only
# ---------------------------------------------------------------------------

# fill: t[i, j] = 0 (i, j); matmul: t[i, j] += x[i, k] * w[k, j] (i, j, k)
FILL = [Access("t", (0, 1), True)]
MATMUL = [
    Access("x", (0, 2), False),
    Access("w", (2, 1), False),
    Access("t", (0, 1), False),
    Access("t", (0, 1), True),
]
# add: t2[i, j] = t[i, j] + b[j] (i, j)
ADD = [Access("t", (0, 1), False), Access("b", (1,), False), Access("t2", (0, 1), True)]


def test_rule_accepts_producer_and_consumer_of_the_same_elements():
    assert can_fuse(FILL, MATMUL, depth=2)
    assert can_fuse(MATMUL, ADD, depth=2)
    assert can_fuse(MATMUL, ADD, depth=1)


def test_rule_accepts_independent_nests():
    assert can_fuse(ADD, [Access("u", (0, 1), True)], depth=2)


def test_rule_rejects_a_transposed_consumer():
    transposed = [Access("t", (1, 0), False), Access("out", (0, 1), True)]
    assert not can_fuse(FILL, transposed, depth=2)
    assert not can_fuse(FILL, transposed, depth=1)


def test_rule_needs_every_common_loop():
    # t[i] written for every j, read by (i, j): iteration (i, j) would read
    # before (i, j + 1) has written
    rows = [Access("t", (0,), True)]
    reads = [Access("t", (0,), False), Access("out", (0, 1), True)]
    assert not can_fuse(rows, reads, depth=2)
    assert can_fuse(rows, reads, depth=1)


def test_rule_fuses_two_layers_on_the_batch_loop():
    # layer 1 writes t4[i, j]; layer 2 reads t4[i, k] inside (i, j', k)
    layer1 = [Access("t4", (0, 1), True)]
    layer2 = [Access("t4", (0, 2), False), Access("t5", (0, 1), True)]
    assert can_fuse(layer1, layer2, depth=1)
    assert not can_fuse(layer1, layer2, depth=2)


# ---------------------------------------------------------------------------
# The frontend's loops
# ---------------------------------------------------------------------------


def test_broadcast_is_fused_into_the_add_and_its_buffer_disappears():
    module = transform(emit(lambda x, b: x + b, (8, 16), (16,)), "algeco-fuse-loops")
    assert len(outermost_loops(module)) == 1
    assert allocs(module) == []  # the broadcast copy %t0 is gone
    (store,) = stores(module)
    assert store.memref.name_hint == "out"
    assert "memref.load %p1[%j]" in to_text(module)  # the bias, read directly


def test_a_layer_becomes_one_nest_with_the_matmul_inside():
    module = transform(
        emit(lambda x, w, b: relu(x @ w + b), (8, 4), (4, 16), (16,)),
        "algeco-fuse-loops",
    )
    assert len(outermost_loops(module)) == 1
    assert chain(module) == ["i", "j", "k"]
    # zero-fill and epilogue around the k loop, in the j loop's body
    j_body = outermost_loops(module)[0].body.block.first_op.body.block
    kinds = [type(op).__name__ for op in j_body.ops]
    assert kinds[0] == "StoreOp" and "ForOp" in kinds and kinds[-2] == "StoreOp"
    # only the accumulator %t0 is left: %t1, %t2, %t3 were forwarded
    assert [a.memref.name_hint for a in allocs(module)] == ["t0"]
    text = to_text(module)
    assert "arith.maximumf" in text and "memref.load %zero[]" in text


def test_two_layers_are_fused_on_the_batch_loop():
    module = transform(
        emit(lambda x, w1, b1, w2, b2: relu(x @ w1 + b1) @ w2 + b2,
             (8, 4), (4, 16), (16,), (16, 10), (10,)),
        "algeco-fuse-loops",
    )  # fmt: skip
    assert len(outermost_loops(module)) == 1
    assert chain(module) == ["i"]  # several nests inside the i loop
    assert [a.memref.name_hint for a in allocs(module)] == ["t0", "t4", "t5"]


def test_pass_is_idempotent():
    module = transform(emit(lambda x, w: x @ w, (8, 4), (4, 16)), "algeco-fuse-loops")
    before = to_text(module)
    assert to_text(transform(module, "algeco-fuse-loops")) == before


# ---------------------------------------------------------------------------
# Hand-written nests
# ---------------------------------------------------------------------------

TWO_ARGS = "%a: memref<4xf32>, %t: memref<4xf32>, %out: memref<4xf32>"


def unchanged(body, args):
    module = parse(body, args)
    before = to_text(module)
    return to_text(transform(module, "algeco-fuse-loops")) == before


def test_different_bounds_are_not_fused():
    assert unchanged(
        """
      scf.for %i = %c0 to %c4 step %c1 {
        %v = memref.load %a[%i] : memref<4xf32>
        memref.store %v, %t[%i] : memref<4xf32>
      }
      scf.for %i = %c0 to %c3 step %c1 {
        %v = memref.load %t[%i] : memref<4xf32>
        memref.store %v, %out[%i] : memref<4xf32>
      }""",
        TWO_ARGS,
    )


def test_consumer_reading_other_iterations_is_not_fused():
    # out[i, j] = t[j, i]: iteration (i, j) needs what (j, i) wrote
    assert unchanged(
        """
      %t = memref.alloc() : memref<4x4xf32>
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %v = memref.load %a[%i, %j] : memref<4x4xf32>
          memref.store %v, %t[%i, %j] : memref<4x4xf32>
        }
      }
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %v = memref.load %t[%j, %i] : memref<4x4xf32>
          memref.store %v, %out[%i, %j] : memref<4x4xf32>
        }
      }
      memref.dealloc %t : memref<4x4xf32>""",
        "%a: memref<4x4xf32>, %out: memref<4x4xf32>",
    )


def test_an_operation_between_the_nests_blocks_fusion():
    assert unchanged(
        """
      scf.for %i = %c0 to %c4 step %c1 {
        %v = memref.load %a[%i] : memref<4xf32>
        memref.store %v, %t[%i] : memref<4xf32>
      }
      memref.store %zero, %t[%c0] : memref<4xf32>
      scf.for %i = %c0 to %c4 step %c1 {
        %v = memref.load %t[%i] : memref<4xf32>
        memref.store %v, %out[%i] : memref<4xf32>
      }""",
        TWO_ARGS,
    )


def test_intermediate_that_belongs_to_the_caller_keeps_its_stores():
    # %t is an argument: the caller reads it after the call
    module = parse(
        """
      scf.for %i = %c0 to %c4 step %c1 {
        %v = memref.load %a[%i] : memref<4xf32>
        %w = arith.addf %v, %v : f32
        memref.store %w, %t[%i] : memref<4xf32>
      }
      scf.for %i = %c0 to %c4 step %c1 {
        %v = memref.load %t[%i] : memref<4xf32>
        memref.store %v, %out[%i] : memref<4xf32>
      }""",
        TWO_ARGS,
    )
    transform(module, "algeco-fuse-loops")
    assert len(outermost_loops(module)) == 1  # fused, and the load forwarded,
    assert {s.memref.name_hint for s in stores(module)} == {"t", "out"}  # but kept


def test_intermediate_with_another_reader_is_kept():
    module = parse(
        """
      %t = memref.alloc() : memref<4xf32>
      scf.for %i = %c0 to %c4 step %c1 {
        %v = memref.load %a[%i] : memref<4xf32>
        memref.store %v, %t[%i] : memref<4xf32>
      }
      scf.for %i = %c0 to %c4 step %c1 {
        %v = memref.load %t[%i] : memref<4xf32>
        memref.store %v, %out[%i] : memref<4xf32>
      }
      scf.for %i = %c0 to %c2 step %c1 {
        %v = memref.load %t[%i] : memref<4xf32>
        memref.store %v, %a[%i] : memref<4xf32>
      }
      memref.dealloc %t : memref<4xf32>""",
        "%a: memref<4xf32>, %out: memref<4xf32>",
    )
    transform(module, "algeco-fuse-loops")
    assert len(outermost_loops(module)) == 2  # the third nest has other bounds
    assert len(allocs(module)) == 1 and len(stores(module)) == 3


# ---------------------------------------------------------------------------
# End to end: results stay bit-identical
# ---------------------------------------------------------------------------


def mlp(x, w1, b1, w2, b2):
    return algeco.relu(x @ w1 + b1) @ w2 + b2


@pytest.mark.parametrize(
    "passes",
    [
        "algeco-fuse-loops",
        "algeco-fuse-loops,algeco-interchange,algeco-tile{sizes=8,8,8}",
        "algeco-interchange,algeco-tile{sizes=8,8,8},algeco-fuse-loops",
        "algeco-interchange,algeco-fuse-loops",
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


def test_pipeline_order_decides_what_fires():
    layer = lambda x, w, b: relu(x @ w + b)  # noqa: E731
    shapes = [(8, 4), (4, 16), (16,)]
    # fusion first: one imperfect nest, nothing left for tiling to do
    first = transform(
        emit(layer, *shapes), "algeco-fuse-loops,algeco-tile{sizes=4,8,2}"
    )
    assert len(outermost_loops(first)) == 1 and "%ii" not in to_text(first)
    # tiling first: the matmul is tiled, and only the elementwise nests fuse
    last = transform(emit(layer, *shapes), "algeco-tile{sizes=4,8,2},algeco-fuse-loops")
    assert [chain(last, n)[:2] for n in range(len(outermost_loops(last)))] == [
        ["i", "j"], ["ii", "jj"], ["i", "j"]
    ]  # fmt: skip
