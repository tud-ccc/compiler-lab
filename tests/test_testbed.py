import numpy as np
import pytest

import algeco
from algeco.testbed import (
    Stage,
    Task,
    compare,
    main,
    pass_effects,
    report,
    run,
    with_llvm_vectorizer,
)


def layer(x, w, b):
    return algeco.relu(x @ w + b)


def numpy_layer(x, w, b):
    return np.maximum(x @ w + b, 0)


def inputs(batch):
    rng = np.random.default_rng(0)
    return [
        rng.standard_normal(s, dtype=np.float32) for s in [(batch, 8), (8, 16), (16,)]
    ]


TASK = Task(
    title="test task",
    model=layer,
    numpy_model=numpy_layer,
    inputs=inputs,
    stages=[
        Stage("baseline", ""),
        Stage("interchange", "algeco-interchange"),
        Stage("twice", "algeco-interchange,algeco-interchange"),
    ],
)


def test_stages_report_pass_effects_and_results(tmp_path):
    results = run(TASK, batch=4, out=tmp_path, repeat=1)
    assert [r.status for r in results] == [
        "reference",
        "bit-identical",
        "bit-identical",
    ]
    assert results[0].changed == []
    assert results[1].changed == [("algeco-interchange", True)]
    # interchange is idempotent: the second run leaves the IR unchanged
    assert results[2].changed == [
        ("algeco-interchange", True),
        ("algeco-interchange", False),
    ]
    assert all(r.seconds > 0 for r in results)
    assert (tmp_path / "interchange").is_dir()


def test_a_failing_stage_does_not_stop_the_others(tmp_path):
    stages = [Stage("baseline", ""), Stage("bad", "no-such-pass"), Stage("ok", "")]
    results = run(TASK, batch=2, out=tmp_path, stages=stages, repeat=1)
    assert [r.status for r in results] == ["reference", "FAILED", "bit-identical"]
    assert "no-such-pass" in results[1].detail
    assert "FAILED" in report(results, None)


def test_compare():
    a = np.arange(4, dtype=np.float32)
    assert compare(a, a.copy(), exact=True)[0] == "bit-identical"
    nudged = a + np.float32(1e-6)
    assert compare(nudged, a, exact=True)[0] == "WRONG"
    assert compare(nudged, a, exact=False)[0] == "close"
    assert compare(a + 1, a, exact=False)[0] == "WRONG"
    assert compare(a[:2], a, exact=True)[0] == "WRONG"


def test_pass_effects_of_a_dump(tmp_path):
    for name, text in [
        ("00-expr.txt", "tree"),
        ("01-emitted.mlir", "A"),
        ("02-p1.mlir", "B"),
        ("03-p2.mlir", "B"),
        ("04-llvm-dialect.mlir", "C"),
    ]:
        (tmp_path / name).write_text(text)
    assert pass_effects(tmp_path) == [("p1", True), ("p2", False)]


@pytest.mark.parametrize(
    "argv, code",
    [(["--repeat", "1"], 0), (["--passes", "no-such-pass", "--repeat", "1"], 1)],
)
def test_main_exit_code(argv, code, tmp_path, capsys):
    assert main(TASK, ["--batch", "2", *argv], out=tmp_path) == code
    assert "test task" in capsys.readouterr().out


def test_llvm_vectorizer_stages(tmp_path):
    stages = with_llvm_vectorizer(TASK.stages[:2])
    assert [(s.name, s.llvm_vectorize) for s in stages] == [
        ("baseline", False),
        ("interchange", False),
        ("baseline, LLVM vectorizer", True),
        ("interchange, LLVM vectorizer", True),
    ]
    results = run(TASK, batch=4, out=tmp_path, stages=stages, repeat=1)
    # LLVM vectorizes without reordering sums (strict floating point)
    assert [r.status for r in results[1:]] == ["bit-identical"] * 3
    log = (results[3].dump / "compile.log").read_text()
    assert "llvm_vectorize: True" in log
