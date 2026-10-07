"""The `algeco` command line driver.

Examples::

    algeco input.mlir -p algeco-interchange
    algeco input.mlir --emit=llvm -o out.ll
    algeco input.mlir --emit=obj --target=riscv64-unknown-linux-gnu -o out.o
"""

import argparse
import sys

from algeco.backend import compile_llvm_ir
from algeco.context import make_context
from algeco.pipeline import parse_module, run_passes, to_llvm_ir, to_text
from algeco.transforms import get_all_passes


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="algeco", description=__doc__.splitlines()[0])
    p.add_argument("input", nargs="?", default="-", help="MLIR input file (- = stdin)")
    p.add_argument("-o", "--output", default="-", help="output file (- = stdout)")
    p.add_argument(
        "-p", "--passes", default="", help="comma-separated xDSL pass pipeline"
    )
    p.add_argument(
        "--emit",
        choices=["mlir", "llvm", "asm", "obj"],
        default="mlir",
        help="what to produce after running the middle-end passes",
    )
    p.add_argument("--target", default=None, help="target triple for --emit=asm/obj")
    p.add_argument(
        "--llvm-vectorize",
        action="store_true",
        help="enable LLVM's own vectorizers (off by default: vectorizing is our job)",
    )
    p.add_argument(
        "--list-passes", action="store_true", help="list available passes and exit"
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list_passes:
        for name in sorted(get_all_passes()):
            print(name)
        return 0

    src = sys.stdin.read() if args.input == "-" else open(args.input).read()
    ctx = make_context()
    module = parse_module(ctx, src, args.input)

    run_passes(ctx, module, args.passes)

    if args.emit == "mlir":
        out: str | bytes = to_text(module)
    else:
        llvm_ir = to_llvm_ir(module)
        out = (
            llvm_ir
            if args.emit == "llvm"
            else compile_llvm_ir(
                llvm_ir,
                kind=args.emit,
                target=args.target,
                llvm_vectorize=args.llvm_vectorize,
            )
        )

    if args.output == "-":
        if isinstance(out, bytes):
            sys.stdout.buffer.write(out)
        else:
            sys.stdout.write(out)
    else:
        mode = "wb" if isinstance(out, bytes) else "w"
        with open(args.output, mode) as f:
            f.write(out)
    return 0
