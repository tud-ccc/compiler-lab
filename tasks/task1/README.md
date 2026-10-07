# Task 1: loop interchange

You finish the loop interchange pass: the cost model that chooses the best loop
order, and the rewrite that rebuilds a loop nest in that order. The handout
explains the details.

## Run the testbed

```
pixi run task1                     # baseline vs. interchange
pixi run task1 --llvm-vectorize    # + both again, with LLVM's vectorizers
pixi run task1 --passes 'algeco-interchange{order=2,0,1}'   # force an order
```

The testbed compiles the MLP without passes (`baseline`) and with
`algeco-interchange`, and reports whether your pass changed the IR, whether the
result is still **bit-identical** to the baseline, and the speedup. If the pass
says `no`, it didn't fire: either it isn't implemented yet, or its
preconditions didn't hold.

The IR after every pass is in `tasks/task1/out/<stage>/mlp_<shapes>/`:

| File | Content |
|---|---|
| `01-emitted.mlir` | the loops emitted by the frontend |
| `02-algeco-interchange.mlir` | after your pass |
| `NN-llvm.ll`, `NN-asm.s` | LLVM IR and assembly |
| `compile.log` | what ran, how long it took, and errors |

## Your code

- `src/algeco/transforms/interchange.py`: section 3 (`cache_lines_per_iteration`,
  `choose_order`) and section 4 (`interchange`). Sections 1 and 2 and the pass
  itself are provided.
- Tests: `tests/test_interchange.py`, `tests/filecheck/transforms/interchange.mlir`

```
pixi run test tests/test_interchange.py
pixi run filecheck
```

## Also useful

- `src/algeco/frontend/emit_loops.py`: what the emitted loops look like, and what
  your pass may rely on (perfect nests, fresh buffers, strict floating point)
- `tests/test_frontend.py`: `matmul_in_loop_order`, a NumPy reference that adds
  every sum in loop order, rounding after each step
