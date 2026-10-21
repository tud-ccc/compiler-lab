"""Helpers shared by the tests of the loop passes (Task 1)."""

from xdsl.dialects import scf
from xdsl.parser import Parser

from algeco.context import make_context
from algeco.frontend import Var
from algeco.frontend.emit_loops import LoopEmitter
from algeco.pipeline import run_passes
from algeco.transforms.interchange import perfect_nest


def emit(build, *shapes):
    """The loops of `build(*params)` for parameters of the given shapes."""
    params = [Var(f"p{n}", shape) for n, shape in enumerate(shapes)]
    return LoopEmitter().emit_module("f", params, build(*params))


def transform(module, passes):
    run_passes(make_context(), module, passes)
    module.verify()
    return module


def parse(body, args="%a: memref<4x4xf32>, %out: memref<4x4xf32>"):
    """A function with the usual constants, around a hand-written `body`."""
    src = f"""
    func.func @f({args}) {{
      %zero = arith.constant 0.0 : f32
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      {body}
      func.return
    }}"""
    return Parser(make_context(), src).parse_module()


def outermost_loops(module):
    return [
        op
        for op in module.walk()
        if isinstance(op, scf.ForOp) and not isinstance(op.parent_op(), scf.ForOp)
    ]


def nests(module):
    """The loop variable names of every outermost perfect nest, e.g. ['i', 'k', 'j']."""
    result = []
    for op in outermost_loops(module):
        nest = perfect_nest(op) or [op]
        result.append([loop.body.block.args[0].name_hint for loop in nest])
    return result


# Nests that none of the loop passes may touch.
LEFT_ALONE = {
    "imperfect nest": """
      scf.for %i = %c0 to %c4 step %c1 {
        memref.store %zero, %out[%i, %c0] : memref<4x4xf32>
        scf.for %j = %c0 to %c4 step %c1 {
          %v = memref.load %a[%j, %i] : memref<4x4xf32>
          memref.store %v, %out[%i, %j] : memref<4x4xf32>
        }
      }""",
    "computed index": """
      scf.for %i = %c0 to %c3 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %i1 = arith.addi %i, %c1 : index
          %v = memref.load %a[%j, %i1] : memref<4x4xf32>
          memref.store %v, %out[%i, %j] : memref<4x4xf32>
        }
      }""",
    "in-place transpose": """
      scf.for %i = %c0 to %c4 step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %v = memref.load %out[%j, %i] : memref<4x4xf32>
          memref.store %v, %out[%i, %j] : memref<4x4xf32>
        }
      }""",
    "non-constant bound": """
      %n = memref.dim %a, %c0 : memref<4x4xf32>
      scf.for %i = %c0 to %n step %c1 {
        scf.for %j = %c0 to %c4 step %c1 {
          %v = memref.load %a[%j, %i] : memref<4x4xf32>
          memref.store %v, %out[%i, %j] : memref<4x4xf32>
        }
      }""",
    "loop-carried value": """
      scf.for %i = %c0 to %c4 step %c1 {
        %s = scf.for %j = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> (f32) {
          %v = memref.load %a[%j, %i] : memref<4x4xf32>
          %acc2 = arith.addf %acc, %v : f32
          scf.yield %acc2 : f32
        }
        memref.store %s, %out[%i, %c0] : memref<4x4xf32>
      }""",
}
