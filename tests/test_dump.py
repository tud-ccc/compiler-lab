import numpy as np
import pytest

import algeco


def test_dump_writes_every_stage(tmp_path):
    @algeco.jit(passes="algeco-interchange", dump=tmp_path)
    def layer(x, w, b):
        return algeco.relu(x @ w + b)

    args = [np.ones(s, dtype=np.float32) for s in [(4, 8), (8, 16), (16,)]]
    np.testing.assert_allclose(layer(*args), np.full((4, 16), 9.0))

    directory = tmp_path / "layer_4x8_8x16_16"
    files = sorted(p.name for p in directory.iterdir())
    assert files == [
        "00-expr.txt",
        "01-emitted.mlir",
        "02-algeco-interchange.mlir",
        "03-llvm-dialect.mlir",
        "04-llvm.ll",
        "05-asm.s",
        "compile.log",
        "kernel.so",
    ]
    assert "scf.for" in (directory / "02-algeco-interchange.mlir").read_text()
    log = (directory / "compile.log").read_text()
    assert "passes: algeco-interchange" in log
    assert "05-asm.s" in log


def test_dump_logs_failures(tmp_path):
    @algeco.jit(passes="no-such-pass", dump=tmp_path)
    def f(x):
        return algeco.relu(x)

    with pytest.raises(ValueError, match="no-such-pass"):
        f(np.ones((2, 2), dtype=np.float32))
    log = (tmp_path / "f_2x2" / "compile.log").read_text()
    assert "FAILED" in log
    assert (tmp_path / "f_2x2" / "01-emitted.mlir").exists()
