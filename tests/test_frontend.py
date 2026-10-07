import numpy as np
import pytest

import algeco
from algeco.frontend import Add, Broadcast, Const, MatMul, Maximum, Var

rng = np.random.default_rng(0)


def rand(*shape):
    return rng.standard_normal(shape, dtype=np.float32)


# ---------------------------------------------------------------------------
# Expression trees
# ---------------------------------------------------------------------------


def test_operators_build_tree():
    x, w, b = Var("x", (4, 3)), Var("w", (3, 5)), Var("b", (5,))
    e = algeco.relu(x @ w + b)
    assert isinstance(e, Maximum)
    assert isinstance(e.lhs, Add) and isinstance(e.lhs.lhs, MatMul)
    assert e.shape == (4, 5)
    assert str(e) == "max(((x @ w) + broadcast(b, (4, 5))), broadcast(zero, (4, 5)))"


def test_relu_broadcasts_a_zero_constant():
    x = Var("x", (4, 5))
    e = algeco.relu(x)
    assert isinstance(e, Maximum) and e.lhs is x
    assert isinstance(e.rhs, Broadcast) and e.rhs.shape == (4, 5)
    zero = e.rhs.operand
    assert isinstance(zero, Const) and zero.shape == () and zero.value == 0


def test_add_makes_broadcasting_explicit():
    x, b = Var("x", (4, 5)), Var("b", (5,))
    e = x + b
    assert isinstance(e.rhs, Broadcast) and e.rhs.operand is b
    assert e.lhs is x  # already the right shape
    assert (x + x).rhs is x  # equal shapes: no broadcast
    row, col = Var("row", (1, 3)), Var("col", (2, 1))
    e = row + col  # both sides repeat
    assert (e.shape, e.lhs.shape, e.rhs.shape) == ((2, 3), (2, 3), (2, 3))


def test_materialized_broadcast_matches_numpy():
    b, w = rand(3), rand(3, 4)

    @algeco.jit
    def f(b, w):
        return algeco.broadcast_to(b, (2, 3)) @ w

    @algeco.jit
    def g(b):
        return algeco.broadcast_to(b, (2, 3))

    expected = np.broadcast_to(b, (2, 3)) @ w
    np.testing.assert_allclose(f(b, w), expected, rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(g(b), np.broadcast_to(b, (2, 3)))


def test_broadcast_to():
    b = Var("b", (5,))
    assert algeco.broadcast_to(b, (5,)) is b
    e = algeco.broadcast_to(algeco.broadcast_to(b, (1, 5)), (4, 5))
    assert isinstance(e, Broadcast) and e.operand is b and e.shape == (4, 5)
    assert isinstance(np.broadcast_to(b, (4, 5)), Broadcast)
    with pytest.raises(ValueError, match="cannot broadcast shape"):
        algeco.broadcast_to(Var("a", (2, 3)), (3,))


def test_jit_argument_errors():
    @algeco.jit
    def f(a, b):
        return a + b

    with pytest.raises(TypeError, match="takes 2 arguments"):
        f(rand(2))

    @algeco.jit
    def not_an_expression(a):
        return 1.0

    with pytest.raises(TypeError, match="must return an algeco expression"):
        not_an_expression(rand(2))


def test_shape_errors():
    with pytest.raises(ValueError, match="matmul shape mismatch"):
        Var("a", (2, 3)) @ Var("b", (4, 5))
    with pytest.raises(ValueError, match="do not broadcast"):
        Var("a", (2, 3)) + Var("b", (4,))


# ---------------------------------------------------------------------------
# Compiled code matches NumPy
# ---------------------------------------------------------------------------


@algeco.jit
def mlp(x, w1, b1, w2, b2):
    return algeco.relu(x @ w1 + b1) @ w2 + b2


def test_mlp_matches_numpy():
    x, w1, b1, w2, b2 = rand(8, 64), rand(64, 32), rand(32), rand(32, 10), rand(10)
    expected = np.maximum(x @ w1 + b1, 0) @ w2 + b2
    np.testing.assert_allclose(mlp(x, w1, b1, w2, b2), expected, rtol=1e-4, atol=1e-4)


def test_recompiles_per_shape():
    for batch in (1, 3):
        args = rand(batch, 4), rand(4, 3), rand(3), rand(3, 2), rand(2)
        assert mlp(*args).shape == (batch, 2)
    assert {shapes[0] for shapes in mlp.cache} >= {(1, 4), (3, 4)}


@pytest.mark.parametrize(
    "fn, reference",
    [
        (lambda a, b: a + b, lambda a, b: a + b),
        (lambda a, b: algeco.relu(a + b), lambda a, b: np.maximum(a + b, 0)),
        (lambda a, b: algeco.maximum(a, b), np.maximum),
        (lambda a, b: a, lambda a, b: a),  # returns an input: copied into %out
    ],
)
def test_elementwise(fn, reference):
    def f(a, b):
        return fn(a, b)

    a, b = rand(3, 4), rand(1, 4)  # b broadcasts along the rows
    np.testing.assert_allclose(algeco.jit(f)(a, b), reference(a, b), rtol=1e-6)


def test_maximum_matches_np_maximum_except_for_signed_zeros():
    # Both propagate NaN. On a tie between -0 and +0, arith.maximumf (IEEE
    # 754-2019 `maximum`) returns +0, while np.maximum returns its second operand.
    a = np.array([np.nan, 1.0, -2.0, np.inf, -0.0, 0.0], np.float32)
    b = np.array([1.0, np.nan, -3.0, -np.inf, 0.0, -0.0], np.float32)
    bits = algeco.jit(lambda a, b: algeco.maximum(a, b))(a, b).view(np.uint32)
    expected = np.maximum(a, b).view(np.uint32)
    np.testing.assert_array_equal(bits[:4], expected[:4])
    assert list(bits[4:]) == [0, 0]  # +0, +0
    assert list(expected[4:]) == [0, 0x80000000]  # +0, -0

    # relu(x) = max(x, +0) agrees: NumPy returns the +0 on the tie as well.
    relu = algeco.jit(lambda a: algeco.relu(a))
    np.testing.assert_array_equal(
        relu(a).view(np.uint32), np.maximum(a, 0).view(np.uint32)
    )


def test_relus_share_one_zero_constant():
    f = algeco.jit(lambda x: algeco.relu(algeco.relu(x) + 1.0))
    text = f.mlir(rand(2, 3))
    assert text.count('sym_name = "cst_zero"') == 1


def test_llvm_vectorizers_are_off_by_default():
    # Vectorizing is the students' job (Task 3): LLVM's own vectorizers would
    # hide what their passes contribute.
    assert algeco.jit(lambda a: a).llvm_vectorize is False
    assert algeco.jit(passes="")(lambda a: a).llvm_vectorize is False


def test_with_llvm_vectorizers():
    @algeco.jit(llvm_vectorize=True)
    def f(a, b):
        return a @ b

    a, b = rand(5, 7), rand(7, 3)
    np.testing.assert_allclose(f(a, b), a @ b, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["free", "malloc", "memcpy"])
def test_function_names_do_not_clash_with_libc(name):
    # The compiled code calls malloc/free for intermediate buffers, so the
    # function's own symbol must not be called `free` or `malloc`.
    def f(a, b):
        return algeco.relu(a @ b)

    f.__name__ = name
    a, b = rand(3, 4), rand(4, 2)
    np.testing.assert_allclose(
        algeco.jit(f)(a, b), np.maximum(a @ b, 0), rtol=1e-5, atol=1e-5
    )


def test_lambda_gets_a_valid_symbol():
    f = algeco.jit(lambda a: algeco.relu(a))
    assert "func.func @algeco__lambda_(" in str(f.emit(rand(2)))


# ---------------------------------------------------------------------------
# Constants: NumPy arrays and numbers as operands
# ---------------------------------------------------------------------------


def test_arrays_and_numbers_become_constants():
    x = Var("x", (2, 3))
    w, b = rand(3, 4), rand(4)
    for e in (x @ w, rand(5, 2) @ x):  # array on either side
        assert isinstance(e, MatMul)
        assert any(isinstance(op, Const) for op in e.operands)
    for e in (x + 1.0, 1.0 + x, b[:3] + x):
        assert isinstance(e, Add)
    assert Const(1.0).shape == ()
    assert Const(np.arange(3)).value.dtype == np.float32  # converted
    assert Const(w).value is w  # float32 and contiguous: not copied


def test_np_maximum_builds_maximum():
    x = Var("x", (2, 3))
    for e in (np.maximum(x, 0), np.maximum(0, x), np.maximum(x, rand(3))):
        assert isinstance(e, Maximum) and e.shape == (2, 3)
        assert x in e.operands
        assert any(isinstance(op, Broadcast) for op in e.operands)
    assert isinstance(np.maximum(x, x), Maximum)
    assert algeco.maximum(x, rand(4, 1, 1)).shape == (4, 2, 3)
    with pytest.raises(ValueError, match="maximum: shapes"):
        algeco.maximum(x, Var("y", (3, 2)))


@pytest.mark.parametrize(
    "build, message",
    [
        (lambda e: np.exp(e), "does not support np.exp"),
        (lambda e: np.dot(e, e), "does not support np.dot"),
        (lambda e: np.add(e, e, out=np.empty((2, 3))), "does not support np.add"),
    ],
)
def test_unsupported_numpy_calls(build, message):
    with pytest.raises(TypeError, match=message):
        build(Var("x", (2, 3)))


def make_mlp(w1, b1, w2, b2):
    @algeco.jit
    def infer(x):
        h = np.maximum(x @ w1 + b1, 0)
        return h @ w2 + b2

    return infer


def mlp_reference(x, w1, b1, w2, b2):
    return np.maximum(x @ w1 + b1, 0) @ w2 + b2


def test_captured_weights_are_compiled_in():
    weights = rand(8, 6), rand(6), rand(6, 3), rand(3)
    infer = make_mlp(*weights)
    for batch in (1, 5):
        x = rand(batch, 8)
        np.testing.assert_allclose(
            infer(x), mlp_reference(x, *weights), rtol=1e-4, atol=1e-4
        )
    text = infer.mlir(rand(1, 8))
    assert "func.func @algeco_infer(%x: memref<1x8xf32>, %out: memref<1x3xf32>)" in text
    for name in ("w1", "b1", "w2", "b2"):
        assert f"%{name} = memref.get_global @cst_{name}" in text


def test_captured_weights_are_frozen_at_compile_time():
    w1, b1, w2, b2 = rand(8, 6), rand(6), rand(6, 3), rand(3)
    infer = make_mlp(w1, b1, w2, b2)
    x = rand(2, 8)
    before = infer(x)
    w1 += 1.0  # compiled code does not see this
    np.testing.assert_array_equal(infer(x), before)
    # a new specialization compiles the new values
    np.testing.assert_allclose(
        make_mlp(w1, b1, w2, b2)(x),
        mlp_reference(x, w1, b1, w2, b2),
        rtol=1e-4,
        atol=1e-4,
    )


def test_mixing_parameters_and_constants():
    w = rand(4, 3)

    @algeco.jit
    def f(x, b):
        return x @ w + b + 0.5

    x, b = rand(2, 4), rand(3)
    np.testing.assert_allclose(f(x, b), x @ w + b + 0.5, rtol=1e-5, atol=1e-5)
    text = f.mlir(x, b)
    assert "@cst_w" in text and "%b: memref<3xf32>" in text


GLOBAL_WEIGHTS = rand(3, 3)


def test_global_arrays_are_constants_too():
    @algeco.jit
    def f(x):
        return x @ GLOBAL_WEIGHTS @ GLOBAL_WEIGHTS  # one array used twice

    x = rand(2, 3)
    np.testing.assert_allclose(
        f(x), x @ GLOBAL_WEIGHTS @ GLOBAL_WEIGHTS, rtol=1e-5, atol=1e-5
    )
    text = f.mlir(x)
    assert text.count('"memref.global"') == 1  # equal constants share one global
    assert "@cst_GLOBAL_WEIGHTS" in text


def test_constant_names_do_not_clash_with_libc():
    # The lowered code calls malloc/free for the intermediate buffer.
    malloc, free = rand(3, 2), rand(2)

    @algeco.jit
    def f(x):
        return np.maximum(x @ malloc + free, 0)

    x = rand(4, 3)
    np.testing.assert_allclose(
        f(x), np.maximum(x @ malloc + free, 0), rtol=1e-5, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Strict floating point: sums are computed in loop order
# ---------------------------------------------------------------------------


def matmul_in_loop_order(a, b):
    """c[i, j] = (((0 + a[i,0] b[0,j]) + a[i,1] b[1,j]) + ...), rounded after each step.

    This is the order of the emitted `k` loop. Interchange and tiling keep it for
    every element, so transformed programs must match this bit for bit.
    """
    c = np.zeros((a.shape[0], b.shape[1]), np.float32)
    for k in range(a.shape[1]):
        c = c + a[:, k : k + 1] * b[k : k + 1, :]  # float32 multiply, then add
    return c


@pytest.mark.parametrize("llvm_vectorize", [True, False])
def test_mlp_is_bit_identical_to_loop_order(llvm_vectorize):
    @algeco.jit(llvm_vectorize=llvm_vectorize)
    def f(x, w1, b1, w2, b2):
        return algeco.relu(x @ w1 + b1) @ w2 + b2

    x, w1, b1, w2, b2 = rand(16, 64), rand(64, 32), rand(32), rand(32, 10), rand(10)
    h = np.maximum(matmul_in_loop_order(x, w1) + b1, 0)
    expected = matmul_in_loop_order(h, w2) + b2
    np.testing.assert_array_equal(f(x, w1, b1, w2, b2), expected)
