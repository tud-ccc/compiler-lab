# Task 1: loop optimizations

Three passes on the loops the frontend emits, built on the pass of
the interchange homework:

| Pass | File | You write |
|---|---|---|
| `algeco-interchange` | `src/algeco/transforms/interchange.py` | (the interchange homework) |
| `algeco-strip-mine`, `algeco-tile` | `src/algeco/transforms/tiling.py` | `strip_mine`, `tile` |
| `algeco-fuse-loops` | `src/algeco/transforms/fusion.py` | `can_fuse`, `fuse`, `forward_stores`, `is_dead_intermediate` |

The handout explains the details.

## Run the testbed

```
pixi run task1                     # every stage below
pixi run task1 --llvm-vectorize    # + every stage again, with LLVM's vectorizers
pixi run task1 --batch 1024        # a batch whose buffers don't fit in the cache
pixi run task1 --passes 'algeco-interchange,algeco-tile{sizes=16,16,64}'
```

The testbed compiles the MLP once per stage and reports, for every pass of
the stage, whether it changed the IR, whether the result is still
**bit-identical** to the baseline, and the speedup:

| Stage | Pipeline |
|---|---|
| `baseline` | no passes |
| `interchange` | `algeco-interchange` |
| `strip-mine` | `algeco-interchange,algeco-strip-mine` |
| `tile` | `algeco-interchange,algeco-tile` |
| `fuse` | `algeco-fuse-loops` |
| `fuse, then tile` | `algeco-fuse-loops,algeco-interchange,algeco-tile` |
| `tile, then fuse` | `algeco-interchange,algeco-tile,algeco-fuse-loops` |

A pass that says `no` didn't fire: either it isn't implemented yet, or its
preconditions didn't hold on that IR (the last two stages are about exactly
that). A stage that `FAILED` raised an exception: the message, and
`compile.log` in its dump, say where.

The IR after every pass is in `tasks/task1/out/<stage>/mlp_<shapes>/`:

| File | Content |
|---|---|
| `01-emitted.mlir` | the loops emitted by the frontend |
| `02-algeco-<pass>.mlir`, `03-...` | after each pass of the pipeline |
| `NN-llvm.ll`, `NN-asm.s` | LLVM IR and assembly |
| `compile.log` | what ran, how long it took, and errors |

## Tests

```
pixi run test tests/test_tiling.py
pixi run test tests/test_fusion.py
pixi run filecheck                 # tests/filecheck/transforms/{tiling,fusion}.mlir
```

`tests/loops.py` has the helpers the tests use (emit a model, run passes,
parse a hand-written nest, list the loop names of every nest).

## Also useful

- `src/algeco/frontend/emit_loops.py`: what the emitted loops look like, and what
  your passes may rely on (perfect nests, fresh buffers, strict floating point)
- `tests/test_frontend.py`: `matmul_in_loop_order`, a NumPy reference that adds
  every sum in loop order, rounding after each step
