# RUN: python %s | filecheck %s
# LoopEmitter: one fresh buffer per intermediate result, freed at the end.

from algeco.frontend import Var, relu
from algeco.frontend.emit_loops import LoopEmitter


def emit(name, params, result):
    print(LoopEmitter().emit_module(name, params, result))


x, w1, b1 = Var("x", (2, 4)), Var("w1", (4, 3)), Var("b1", (3,))
w2, b2 = Var("w2", (3, 5)), Var("b2", (5,))
emit("mlp", [x, w1, b1, w2, b2], relu(x @ w1 + b1) @ w2 + b2)

# All constants come first, so every loop nest can use them.
# CHECK-LABEL: func.func @mlp(
# CHECK-NEXT:    %c0_f32 = arith.constant 0.000000e+00 : f32
# CHECK-NEXT:    %c0 = arith.constant 0 : index
# CHECK:         %zero = memref.get_global @cst_zero : memref<f32>
# CHECK:         %t0 = memref.alloc() : memref<2x3xf32>
# CHECK:         memref.store %{{.*}}, %t0[
# The bias is broadcast into a buffer of its own, then added.
# CHECK:         %t1 = memref.alloc() : memref<2x3xf32>
# CHECK:         memref.load %b1[
# CHECK-NEXT:    memref.store %{{.*}}, %t1[
# CHECK:         %t2 = memref.alloc() : memref<2x3xf32>
# CHECK:         memref.load %t0[
# CHECK-NEXT:    memref.load %t1[
# CHECK-NEXT:    arith.addf
# CHECK-NEXT:    memref.store %{{.*}}, %t2[
# So is the zero of the relu.
# CHECK:         %t3 = memref.alloc() : memref<2x3xf32>
# CHECK:         memref.load %zero[]
# CHECK-NEXT:    memref.store %{{.*}}, %t3[
# CHECK:         %t4 = memref.alloc() : memref<2x3xf32>
# CHECK:         memref.load %t2[
# CHECK-NEXT:    memref.load %t3[
# CHECK-NEXT:    arith.maximumf
# CHECK-NEXT:    memref.store %{{.*}}, %t4[
# CHECK:         %t5 = memref.alloc() : memref<2x5xf32>
# CHECK:         %t6 = memref.alloc() : memref<2x5xf32>
# Only the last node (the second bias add) writes into %out.
# CHECK-NOT:     memref.store %{{.*}}, %out[
# CHECK:         memref.load %t5[
# CHECK-NEXT:    memref.load %t6[
# CHECK-NEXT:    arith.addf
# CHECK-NEXT:    memref.store %{{.*}}, %out[
# Buffers are freed in reverse order, right before returning.
# CHECK:         memref.dealloc %t6 : memref<2x5xf32>
# CHECK-NEXT:    memref.dealloc %t5 : memref<2x5xf32>
# CHECK-NEXT:    memref.dealloc %t4 : memref<2x3xf32>
# CHECK-NEXT:    memref.dealloc %t3 : memref<2x3xf32>
# CHECK-NEXT:    memref.dealloc %t2 : memref<2x3xf32>
# CHECK-NEXT:    memref.dealloc %t1 : memref<2x3xf32>
# CHECK-NEXT:    memref.dealloc %t0 : memref<2x3xf32>
# CHECK-NEXT:    func.return

# A subexpression used twice is computed once, into one buffer.
a, b = Var("a", (2, 2)), Var("b", (2, 2))
t = a @ b
emit("shared", [a, b], t + t)
# CHECK-LABEL: func.func @shared(
# CHECK:         %t0 = memref.alloc() : memref<2x2xf32>
# CHECK-COUNT-1: arith.mulf
# CHECK-NOT:     arith.mulf
# CHECK:         %[[X:.+]] = memref.load %t0[%[[I:[a-z0-9_]+]], %[[J:[a-z0-9_]+]]]
# CHECK-NEXT:    %[[Y:.+]] = memref.load %t0[%[[I]], %[[J]]]
# CHECK-NEXT:    arith.addf %[[X]], %[[Y]]
# CHECK-NOT:     memref.alloc

# Returning an input unchanged: copied into the output buffer.
y = Var("y", (2, 3))
emit("identity", [y], y)
# CHECK-LABEL: func.func @identity(%y: memref<2x3xf32>, %out: memref<2x3xf32>) {
# CHECK-NEXT:    "memref.copy"(%y, %out)
# CHECK-NEXT:    func.return
