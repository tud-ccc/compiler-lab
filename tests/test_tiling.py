"""Tests for strip-mining and tiling (Task 1).

- the passes on IR: which loops they split, where the tile loops go, and what
  they must leave alone
- end to end: compiled results stay bit-identical
"""

import numpy as np
import pytest
from loops import LEFT_ALONE, emit, nests, parse, transform

import algeco
from algeco.pipeline import to_text


def matmul():
    return emit(lambda x, w: x @ w, (8, 4), (4, 16))  # fill i, j; matmul i, j, k


# ---------------------------------------------------------------------------
# Strip-mining
# ---------------------------------------------------------------------------


def test_strip_mining_keeps_the_order_of_the_loops():
    module = transform(matmul(), "algeco-strip-mine{sizes=4,8,2}")
    # the 2-deep fill nest doesn't match the three sizes and is left alone
    assert nests(module) == [["i", "j"], ["ii", "i0", "jj", "j0", "kk", "k0"]]


def test_loop_variables_are_computed_in_the_innermost_body():
    module = transform(matmul(), "algeco-strip-mine{sizes=4,8,2}")
    text = to_text(module)
    assert "arith.addi %ii, %i0" in text and "arith.addi %kk, %k0" in text
    assert "memref.load %p0[%i_1, %k]" in text  # the body still uses i, j, k


def test_sizes_that_dont_divide_or_dont_split_are_skipped():
    # 3 doesn't divide 8; 16 is the whole j loop; 1 is no tile
    module = transform(matmul(), "algeco-strip-mine{sizes=3,16,1}")
    assert nests(module) == [["i", "j"], ["i", "j", "k"]]
    module = transform(matmul(), "algeco-strip-mine{sizes=3,8,1}")
    assert nests(module) == [["i", "j"], ["i", "jj", "j0", "k"]]


# ---------------------------------------------------------------------------
# Tiling
# ---------------------------------------------------------------------------


def test_tiling_moves_the_tile_loops_out():
    module = transform(matmul(), "algeco-tile{sizes=4,8,2}")
    assert nests(module) == [["i", "j"], ["ii", "jj", "kk", "i0", "j0", "k0"]]


def test_sizes_apply_by_position_when_the_pass_runs():
    # after interchange the matmul is i, k, j with 8, 4, 16 iterations: a
    # tile of 8 is the whole k loop, so k stays a point loop
    module = transform(matmul(), "algeco-interchange,algeco-tile{sizes=4,8,2}")
    assert nests(module) == [["i", "j"], ["ii", "jj", "i0", "k", "j0"]]


def test_sizes_of_another_depth_tile_other_nests():
    module = transform(matmul(), "algeco-tile{sizes=4,8}")
    assert nests(module) == [["ii", "jj", "i0", "j0"], ["i", "j", "k"]]


def test_tiling_twice_changes_nothing_more():
    once = to_text(transform(matmul(), "algeco-tile{sizes=4,8,2}"))
    twice = to_text(
        transform(matmul(), "algeco-tile{sizes=4,8,2},algeco-tile{sizes=2,2,2,2,2,2}")
    )
    assert twice == once  # the tiled nest has computed indices: left alone


def test_not_fully_permutable_nest_is_not_tiled():
    # out[i, j] = out[i - 1 + 1, j]... a dependence that forbids j outside i:
    # written and read with different loops, so `is_legal` rejects (1, 0)
    body = """
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %v = memref.load %out[%j, %i] : memref<4x4xf32>
          memref.store %v, %out[%i, %j] : memref<4x4xf32>
        }
      }"""
    module = parse(body)
    before = to_text(module)
    assert to_text(transform(module, "algeco-tile{sizes=2,2}")) == before
    # strip-mining alone never changes the order, so it doesn't need legality
    assert nests(transform(module, "algeco-strip-mine{sizes=2,2}")) == [
        ["ii", "i0", "jj", "j0"]
    ]


@pytest.mark.parametrize("body", LEFT_ALONE.values(), ids=LEFT_ALONE.keys())
@pytest.mark.parametrize(
    "passes", ["algeco-strip-mine{sizes=2,2}", "algeco-tile{sizes=2,2}"]
)
def test_leaves_unsupported_nests_alone(body, passes):
    if "strip-mine" in passes and body == LEFT_ALONE["in-place transpose"]:
        return  # strip-mining is always legal: it does strip-mine this one
    module = parse(body)
    before = to_text(module)
    assert to_text(transform(module, passes)) == before


# ---------------------------------------------------------------------------
# End to end: results stay bit-identical
# ---------------------------------------------------------------------------


def mlp(x, w1, b1, w2, b2):
    return algeco.relu(x @ w1 + b1) @ w2 + b2


@pytest.mark.parametrize(
    "passes",
    [
        "algeco-strip-mine{sizes=8,8,8}",
        "algeco-tile{sizes=8,8,8}",
        "algeco-tile{sizes=4,2,16}",
        "algeco-interchange,algeco-tile{sizes=8,8,8}",
        "algeco-interchange,algeco-tile{sizes=16,2,8}",
    ],
)
def test_mlp_results_are_bit_identical(passes):
    rng = np.random.default_rng(0)
    # layer 1: 16, 24, 32 iterations; layer 2: 16, 10, 24 (10 is not tiled by 8)
    args = [
        rng.standard_normal(s, dtype=np.float32)
        for s in [(16, 32), (32, 24), (24,), (24, 10), (10,)]
    ]
    expected = algeco.jit(mlp)(*args)
    np.testing.assert_array_equal(algeco.jit(mlp, passes=passes)(*args), expected)
