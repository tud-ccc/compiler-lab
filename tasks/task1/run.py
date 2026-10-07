"""Task 1 testbed: loop interchange on the digit MLP.

    pixi run task1                     # baseline vs. interchange
    pixi run task1 --llvm-vectorize    # + both again, with LLVM's vectorizers
    pixi run task1 --passes 'algeco-interchange{order=2,0,1}'   # force an order

See README.md in this directory for where your code and its tests are.
"""

import sys
from pathlib import Path

import numpy as np

import algeco
from algeco.testbed import Stage, Task, main


def mlp(x, w1, b1, w2, b2):
    return algeco.relu(x @ w1 + b1) @ w2 + b2


def numpy_mlp(x, w1, b1, w2, b2):
    return np.maximum(x @ w1 + b1, 0) @ w2 + b2


def inputs(batch: int) -> list[np.ndarray]:
    """Random input and weights of the digit classifier: 64 -> 256 -> 10."""
    rng = np.random.default_rng(0)
    shapes = [(batch, 64), (64, 256), (256,), (256, 10), (10,)]
    return [rng.standard_normal(shape, dtype=np.float32) for shape in shapes]


TASK = Task(
    title="Task 1: loop interchange",
    description="Model: digit MLP, relu(x @ w1 + b1) @ w2 + b2, 64 -> 256 -> 10",
    model=mlp,
    numpy_model=numpy_mlp,
    inputs=inputs,
    stages=[
        Stage("baseline", ""),
        Stage("interchange", "algeco-interchange"),
    ],
    # Loops have strict floating point: interchange keeps the order of every
    # sum, so results must be bit-identical to the baseline.
    exact=True,
)

if __name__ == "__main__":
    sys.exit(main(TASK, out=Path(__file__).parent / "out"))
