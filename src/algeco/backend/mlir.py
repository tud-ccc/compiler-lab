"""Lowering from loops on buffers to LLVM IR using upstream `mlir-opt`.

The backend does no optimization and no bufferization: it only translates
`scf`/`memref`/`arith`/`vector`/`func` to the LLVM dialect. Everything that
makes the code fast has to happen in algeco's own passes before this point.
"""

from xdsl.dialects.builtin import ModuleOp, TensorType

from algeco.backend.tools import ToolError, run_tool

LOWER_TO_LLVM_PIPELINE = ",".join(
    [
        "convert-vector-to-scf",
        "convert-scf-to-cf",
        "expand-strided-metadata",
        "lower-affine",
        "finalize-memref-to-llvm",
        "convert-vector-to-llvm",
        "convert-arith-to-llvm",
        # memref arguments become plain pointers (static shapes only), so the
        # compiled function can be called with NumPy data pointers via ctypes.
        "convert-func-to-llvm{use-bare-ptr-memref-call-conv=1}",
        "convert-cf-to-llvm",
        "reconcile-unrealized-casts",
    ]
)


def check_lowerable(module: ModuleOp) -> None:
    """Reject IR that still works on tensors: bufferization is algeco's job."""
    for op in module.walk():
        for value in (*op.operands, *op.results):
            if isinstance(value.type, TensorType):
                raise ToolError(
                    f"'{op.name}' still uses tensors ({value.type}). The backend only "
                    "accepts buffers (memref): bufferize before lowering to LLVM."
                )


def lower_to_llvm_dialect(mlir_text: str) -> str:
    out = run_tool(
        "mlir-opt",
        [f"--pass-pipeline=builtin.module({LOWER_TO_LLVM_PIPELINE})"],
        mlir_text,
    )
    return out.decode()


def translate_to_llvm_ir(llvm_dialect: str) -> str:
    return run_tool("mlir-translate", ["--mlir-to-llvmir"], llvm_dialect).decode()


def lower_to_llvm_ir(mlir_text: str) -> str:
    return translate_to_llvm_ir(lower_to_llvm_dialect(mlir_text))
