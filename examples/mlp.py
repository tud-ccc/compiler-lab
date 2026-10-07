"""The MLP of the lab, compiled with algeco and compared with NumPy.

pixi run mlp              # check the result, time against NumPy
pixi run mlp --emit       # print the IR emitted by the frontend
"""

import sys
import time
from pathlib import Path

import numpy as np

import algeco

HERE = Path(__file__).parent


def np_reference(x, w1, b1, w2, b2):
    return np.maximum(x @ w1 + b1, 0) @ w2 + b2


@algeco.jit(dump=f"{HERE}/out/dump/digit_mlp")
def mlp_default(x, w1, b1, w2, b2):
    return np_reference(x, w1, b1, w2, b2)


def mlp_staged(w1, b1, w2, b2):
    @algeco.jit(dump=f"{HERE}/out/dump/digit_mlp_staged")
    def infer(x):
        return np_reference(x, w1, b1, w2, b2)

    return infer


def best_time(f, *args, repeat=20):
    f(*args)  # warmup
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        f(*args)
        best = min(best, time.perf_counter() - t0)
    return best


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    batch, d_in, d_hidden, d_out = 64, 64, 256, 10

    args = [
        rng.standard_normal(shape, dtype=np.float32)
        for shape in [
            (batch, d_in),
            (d_in, d_hidden),
            (d_hidden,),
            (d_hidden, d_out),
            (d_out,),
        ]
    ]
    x, weights = args[0], args[1:]

    if "--staged" in sys.argv:
        mlp = mlp_staged(*weights)
    else:

        def mlp(x):
            return mlp_default(x, *weights)

    def reference(x):
        return np_reference(x, *weights)

    np.testing.assert_allclose(mlp(x), reference(x), rtol=1e-4, atol=1e-4)
    print("algeco output matches NumPy")
    for name, f in [("algeco", mlp), ("NumPy", reference)]:
        print(f"{name:>7}: {best_time(f, x) / batch * 1e6:7.2f} µs per image")
