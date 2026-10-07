"""Dumping the IR of every compilation stage, for `algeco.jit(dump=...)`.

A dump directory holds one file per stage, numbered in pipeline order, each
with the output of the stage it is named after::

    00-expr.txt                 the expression tree
    01-emitted.mlir             loops on buffers, from the frontend
    02-algeco-interchange.mlir  after each pass of the pipeline
    ...
    NN-llvm-dialect.mlir        after mlir-opt (LLVM dialect)
    NN-llvm.ll                  after mlir-translate (LLVM IR)
    NN-asm.s                    clang's assembly for the host
    kernel.so                   the library that is loaded
    compile.log                 what ran, and how long each stage took
"""

import shutil
import time
from pathlib import Path


class Dump:
    def __init__(self, directory: Path):
        # The directory belongs to one compilation: remove files of earlier runs,
        # which may have had a different number of stages.
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True)
        self.directory = directory
        self.count = 0
        self.log_file = (directory / "compile.log").open("w")
        self.start = self.last = time.perf_counter()

    def log(self, message: str) -> None:
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def stage(self, name: str, text: str, suffix: str = ".mlir") -> None:
        """Write the output of a stage and log how long the stage took."""
        now = time.perf_counter()
        file = f"{self.count:02d}-{name}{suffix}"
        (self.directory / file).write_text(text)
        self.count += 1
        self.log_time(now, f"{file} ({text.count(chr(10))} lines)")

    def done(self, what: str) -> None:
        """Log the end of a stage that produces no text, e.g. a binary."""
        self.log_time(time.perf_counter(), what)

    def log_time(self, now: float, what: str) -> None:
        self.log(f"{now - self.start:8.3f} s  {now - self.last:+8.3f} s  {what}")
        # Writing the dump does not count towards the next stage.
        self.last = time.perf_counter()

    def close(self) -> None:
        self.log_file.close()
