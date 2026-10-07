from pathlib import Path

import pytest

from algeco.backend.tools import ToolError
from algeco.cli import main

EXAMPLE = Path(__file__).parent.parent / "examples" / "mlp.mlir"


def test_list_passes(capsys):
    assert main(["--list-passes"]) == 0
    assert "algeco-interchange" in capsys.readouterr().out


def test_roundtrip(capsys):
    assert main([str(EXAMPLE)]) == 0
    assert "scf.for" in capsys.readouterr().out


@pytest.mark.parametrize("target", [None, "riscv64"])
def test_emit_asm(capsys, target):
    args = [str(EXAMPLE), "--emit=asm"] + ([f"--target={target}"] if target else [])
    assert main(args) == 0
    assert "mlp" in capsys.readouterr().out


def test_backend_rejects_tensors(tmp_path):
    src = tmp_path / "t.mlir"
    src.write_text(
        "func.func @f(%a: tensor<4xf32>) -> tensor<4xf32> {\n"
        "  func.return %a : tensor<4xf32>\n}\n"
    )
    with pytest.raises(ToolError, match="still uses tensors"):
        main([str(src), "--emit=llvm"])


def test_large_constants_roundtrip_as_resources(tmp_path, capsys):
    import numpy as np

    import algeco

    w = np.arange(64, dtype=np.float32).reshape(8, 8)

    @algeco.jit(dump=tmp_path)
    def f(x):
        return x @ w

    x = np.eye(8, dtype=np.float32)
    np.testing.assert_array_equal(f(x), w)

    # The data of the large constant is out of the way, at the end.
    emitted = tmp_path / "f_8x8" / "01-emitted.mlir"
    text = emitted.read_text()
    assert "initial_value = dense_resource<cst_w>" in text
    assert text.index("func.func") < text.index("dialect_resources")

    # The driver reads it back: same text, same LLVM IR.
    assert main([str(emitted)]) == 0
    assert capsys.readouterr().out == text
    assert main([str(emitted), "--emit=llvm"]) == 0
    llvm_ir = capsys.readouterr().out
    dumped = (emitted.parent / "03-llvm.ll").read_text()

    # Equal up to the first lines (module ID, source file name).
    def body(ir):
        return ir.split("\n", 2)[2]

    assert body(llvm_ir) == body(dumped)


def test_llvm_vectorizers_only_on_request(capsys, tmp_path):
    # An elementwise loop that LLVM vectorizes whenever it is allowed to.
    src = tmp_path / "add.mlir"
    src.write_text(
        "func.func @add(%a: memref<1024xf32>, %out: memref<1024xf32>) {\n"
        "  %c0 = arith.constant 0 : index\n"
        "  %c1 = arith.constant 1 : index\n"
        "  %n = arith.constant 1024 : index\n"
        "  scf.for %i = %c0 to %n step %c1 {\n"
        "    %v = memref.load %a[%i] : memref<1024xf32>\n"
        "    %s = arith.addf %v, %v : f32\n"
        "    memref.store %s, %out[%i] : memref<1024xf32>\n"
        "  }\n"
        "  func.return\n}\n"
    )
    assert main([str(src), "--emit=asm"]) == 0
    default = capsys.readouterr().out
    assert main([str(src), "--emit=asm", "--llvm-vectorize"]) == 0
    vectorized = capsys.readouterr().out
    assert default != vectorized
