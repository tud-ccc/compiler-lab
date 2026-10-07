"""A testbed for the passes of a task (used by `tasks/taskN/run.py`).

Compiles one model with a sequence of pass pipelines ("stages") and reports,
for every stage:

- which passes changed the IR, and which left it unchanged
- whether the result is still correct, compared with the stage without passes
- how fast the compiled code runs, and the speedup over the stage without passes

Every stage is compiled with `dump=`, so the IR after each pass is in
`<out>/<stage>/<model>_<shapes>/` (see `algeco/dump.py`).
"""

import argparse
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import algeco
from algeco.frontend import Expr


@dataclass(frozen=True)
class Stage:
    name: str
    passes: str  # a pass pipeline, e.g. "algeco-interchange,algeco-tile"
    llvm_vectorize: bool = False  # let LLVM's own vectorizers run on the result

    @property
    def directory(self) -> str:
        return re.sub(r"\W+", "-", self.name).strip("-") or "stage"


@dataclass
class Task:
    """What a task's testbed compiles and runs."""

    title: str
    model: Callable[..., Expr]  # an algeco model: parameters -> expression
    numpy_model: Callable[..., np.ndarray]  # the same computation in NumPy
    inputs: Callable[[int], list[np.ndarray]]  # batch size -> arguments
    stages: Sequence[Stage]  # the first one is the reference, usually no passes
    exact: bool = True  # results must be bit-identical to the reference
    description: str = ""


@dataclass
class Result:
    stage: Stage
    status: str  # reference, bit-identical, close, WRONG, FAILED
    detail: str = ""
    changed: list[tuple[str, bool]] | None = None  # (pass, changed the IR?)
    seconds: float | None = None  # per sample
    dump: Path | None = None


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def pass_effects(dump: Path) -> list[tuple[str, bool]]:
    """For each pass in a dump directory: its name, and whether it changed the IR.

    The dump has one numbered file per stage; the passes are the files between
    `NN-emitted.mlir` and `NN-llvm-dialect.mlir`.
    """
    files = sorted(dump.glob("[0-9][0-9]-*.mlir"))
    names = [f.stem[3:] for f in files]
    if "emitted" not in names:
        return []
    start = names.index("emitted")
    end = names.index("llvm-dialect") if "llvm-dialect" in names else len(names)
    effects = []
    for before, after in zip(
        files[start : end - 1], files[start + 1 : end], strict=True
    ):
        effects.append((after.stem[3:], after.read_text() != before.read_text()))
    return effects


def compare(result: np.ndarray, reference: np.ndarray, exact: bool) -> tuple[str, str]:
    """Status and detail of a result compared with the reference."""
    if result.shape != reference.shape:
        return "WRONG", f"shape {result.shape}, expected {reference.shape}"
    if np.array_equal(result, reference):
        return "bit-identical", ""
    error = float(np.max(np.abs(result - reference)))
    if not exact and np.allclose(result, reference, rtol=1e-4, atol=1e-4):
        return "close", f"max error {error:.1e}"
    return "WRONG", f"max error {error:.1e}"


def best_time(f: Callable[[], object], repeat: int) -> float:
    f()  # warm-up
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - start)
    return best


def run_stage(
    task: Task,
    stage: Stage,
    args: list[np.ndarray],
    reference: np.ndarray | None,
    out: Path,
    repeat: int,
) -> tuple[Result, np.ndarray | None]:
    """Compile, check and time one stage. Never raises: errors become FAILED."""
    jitted = algeco.jit(
        task.model,
        passes=stage.passes,
        llvm_vectorize=stage.llvm_vectorize,
        dump=out / stage.directory,
    )
    try:
        output = jitted(*args)
    except Exception as e:
        dumps = sorted((out / stage.directory).glob("*/"))
        dump = dumps[0] if dumps else None
        return Result(stage, "FAILED", f"{type(e).__name__}: {e}", dump=dump), None
    (dump,) = sorted((out / stage.directory).glob("*/"))
    if reference is None:
        status, detail = "reference", ""
    else:
        status, detail = compare(output, reference, task.exact)
    seconds = best_time(lambda: jitted(*args), repeat) / len(args[0])
    return Result(stage, status, detail, pass_effects(dump), seconds, dump), output


def run(
    task: Task,
    batch: int = 64,
    out: Path = Path("out"),
    stages: Sequence[Stage] | None = None,
    repeat: int = 20,
) -> list[Result]:
    """Run all stages; the first stage's output is the reference for the others."""
    args = task.inputs(batch)
    results = []
    reference = None
    for stage in stages if stages is not None else task.stages:
        result, output = run_stage(task, stage, args, reference, out, repeat)
        results.append(result)
        if reference is None:
            if output is None:
                break  # without a reference, nothing else can be checked
            expected = task.numpy_model(*args)
            if not np.allclose(output, expected, rtol=1e-4, atol=1e-4):
                result.status, result.detail = "WRONG", "differs from NumPy"
            reference = output
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def relative(path: Path) -> Path:
    """`path` relative to the current directory, if it is below it."""
    try:
        return path.resolve().relative_to(Path.cwd())
    except ValueError:
        return path


def short(pass_name: str) -> str:
    return pass_name.removeprefix("algeco-")


def report(results: Sequence[Result], numpy_seconds: float | None) -> str:
    base = next((r.seconds for r in results if r.seconds), None)
    rows = [("stage", "changed the IR", "result", "µs/sample", "speedup")]
    for r in results:
        if r.changed:
            changed = ", ".join(
                f"{short(p)} {'yes' if c else 'no'}" for p, c in r.changed
            )
        else:
            changed = "-"
        result = r.status + (
            f" ({r.detail})" if r.detail and r.status != "FAILED" else ""
        )
        time_ = f"{r.seconds * 1e6:.2f}" if r.seconds else "-"
        speedup = f"{base / r.seconds:.1f}x" if base and r.seconds else "-"
        rows.append((r.stage.name, changed, result, time_, speedup))
    if numpy_seconds:
        speedup = f"{base / numpy_seconds:.1f}x" if base else "-"
        rows.append(("NumPy", "-", "-", f"{numpy_seconds * 1e6:.2f}", speedup))

    widths = [max(len(row[c]) for row in rows) for c in range(5)]
    lines = []
    for n, row in enumerate(rows):
        cells = [
            cell.rjust(w) if c >= 3 else cell.ljust(w)
            for c, (cell, w) in enumerate(zip(row, widths, strict=True))
        ]
        lines.append("  " + "  ".join(cells).rstrip())
        if n == 0:
            lines.append("  " + "  ".join("-" * w for w in widths))
    for r in results:
        if r.status == "FAILED":
            lines.append(f"\n  {r.stage.name} FAILED: {r.detail}")
            if r.dump:
                lines.append(f"  see {relative(r.dump / 'compile.log')}")
    return "\n".join(lines)


def with_llvm_vectorizer(stages: Sequence[Stage]) -> list[Stage]:
    """The stages, followed by the same stages compiled with LLVM's vectorizers."""
    vectorized = [
        Stage(f"{s.name}, LLVM vectorizer", s.passes, llvm_vectorize=True)
        for s in stages
    ]
    return [*stages, *vectorized]


def main(task: Task, argv: Sequence[str] | None = None, out: Path | None = None) -> int:
    """Command line of a task script. Returns the exit code."""
    p = argparse.ArgumentParser(description=task.title)
    p.add_argument("--batch", type=int, default=64, help="batch size (default 64)")
    p.add_argument("--passes", help="run this pipeline instead (and the reference)")
    p.add_argument(
        "--llvm-vectorize",
        action="store_true",
        help="also run every stage with LLVM's own vectorizers",
    )
    p.add_argument("--repeat", type=int, default=20, help="timing repetitions")
    args = p.parse_args(argv)

    stages = list(task.stages)
    if args.passes is not None:
        stages = [stages[0], Stage("custom", args.passes)]
    if args.llvm_vectorize:
        stages = with_llvm_vectorizer(stages)
    out = out or Path("out")

    print(task.title)
    if task.description:
        print(task.description)
    print(
        f"batch {args.batch}"
        + (", results must be bit-identical to the baseline" if task.exact else "")
    )
    print()
    width = max(len(s.name) for s in stages)
    for s in stages:
        print(f"  {s.name.ljust(width)}  {s.passes or '(no passes)'}")
    print()

    results = run(task, args.batch, out, stages, args.repeat)
    inputs = task.inputs(args.batch)
    numpy_seconds = (
        best_time(lambda: task.numpy_model(*inputs), args.repeat) / args.batch
    )
    print(report(results, numpy_seconds))
    print(f"\nIR after every pass: {relative(out)}/<stage>/<model>_<shapes>/")
    ok = all(r.status in ("reference", "bit-identical", "close") for r in results)
    return 0 if ok else 1
