# RUN: python %s | filecheck %s
# LoopEmitter: elementwise ops, and broadcasts (copied into a full buffer first).

from algeco.frontend import Var, broadcast_to, relu
from algeco.frontend.emit_loops import LoopEmitter


def emit(name, params, result):
    print(LoopEmitter().emit_module(name, params, result))


# Same shapes: both operands indexed by all loop variables.
a, b = Var("a", (2, 3)), Var("b", (2, 3))
emit("add", [a, b], a + b)
# CHECK-LABEL: func.func @add(
# CHECK:       scf.for %[[I:[a-z0-9_]+]] = %c0 to %c2 step %c1 {
# CHECK-NEXT:    scf.for %[[J:[a-z0-9_]+]] = %c0 to %c3 step %c1 {
# CHECK-NEXT:      %[[X:.+]] = memref.load %a[%[[I]], %[[J]]] : memref<2x3xf32>
# CHECK-NEXT:      %[[Y:.+]] = memref.load %b[%[[I]], %[[J]]] : memref<2x3xf32>
# CHECK-NEXT:      %[[S:.+]] = arith.addf %[[X]], %[[Y]] : f32
# CHECK-NEXT:      memref.store %[[S]], %out[%[[I]], %[[J]]] : memref<2x3xf32>

# A bias vector: first copied to every row (a missing leading dimension is
# indexed by the column only), then added like any operand of the same shape.
x, bias = Var("x", (2, 3)), Var("bias", (3,))
emit("add_bias", [x, bias], x + bias)
# CHECK-LABEL: func.func @add_bias(
# CHECK:       %[[T:.+]] = memref.alloc() : memref<2x3xf32>
# CHECK-NEXT:  scf.for %[[I:[a-z0-9_]+]] = %c0 to %c2 step %c1 {
# CHECK-NEXT:    scf.for %[[J:[a-z0-9_]+]] = %c0 to %c3 step %c1 {
# CHECK-NEXT:      %[[B:.+]] = memref.load %bias[%[[J]]] : memref<3xf32>
# CHECK-NEXT:      memref.store %[[B]], %[[T]][%[[I]], %[[J]]] : memref<2x3xf32>
# CHECK:       scf.for %[[I2:[a-z0-9_]+]] = %c0 to %c2 step %c1 {
# CHECK-NEXT:    scf.for %[[J2:[a-z0-9_]+]] = %c0 to %c3 step %c1 {
# CHECK-NEXT:      memref.load %x[%[[I2]], %[[J2]]]
# CHECK-NEXT:      memref.load %[[T]][%[[I2]], %[[J2]]] : memref<2x3xf32>

# Dimensions of size 1 are repeated: index 0 (operands on either side).
row, col = Var("row", (1, 3)), Var("col", (2, 1))
emit("add_outer", [row, col], row + col)
# CHECK-LABEL: func.func @add_outer(%row: memref<1x3xf32>, %col: memref<2x1xf32>, %out: memref<2x3xf32>)
# CHECK:       scf.for %[[I:[a-z0-9_]+]] = %c0 to %c2 step %c1 {
# CHECK-NEXT:    scf.for %[[J:[a-z0-9_]+]] = %c0 to %c3 step %c1 {
# CHECK-NEXT:      memref.load %row[%c0, %[[J]]]
# CHECK:       scf.for %[[I:[a-z0-9_]+]] = %c0 to %c2 step %c1 {
# CHECK-NEXT:    scf.for %[[J:[a-z0-9_]+]] = %c0 to %c3 step %c1 {
# CHECK-NEXT:      memref.load %col[%[[I]], %c0]

# ReLU: max(x, broadcast(0)). The zero is a 0-d constant, copied to every
# element like a bias, then compared like any operand of the same shape.
v = Var("v", (5,))
emit("relu", [v], relu(v))
# CHECK:       "memref.global"() <{sym_name = "cst_zero", type = memref<f32>, initial_value = dense<0.000000e+00> : tensor<f32>
# CHECK-LABEL: func.func @relu(
# CHECK:       %zero = memref.get_global @cst_zero : memref<f32>
# CHECK:       %[[T:.+]] = memref.alloc() : memref<5xf32>
# CHECK-NEXT:  scf.for %[[I:[a-z0-9_]+]] = %c0 to %c5 step %c1 {
# CHECK-NEXT:    %[[Z:.+]] = memref.load %zero[] : memref<f32>
# CHECK-NEXT:    memref.store %[[Z]], %[[T]][%[[I]]] : memref<5xf32>
# CHECK:       scf.for %[[I:[a-z0-9_]+]] = %c0 to %c5 step %c1 {
# CHECK-NEXT:    %[[X:.+]] = memref.load %v[%[[I]]] : memref<5xf32>
# CHECK-NEXT:    %[[Y:.+]] = memref.load %[[T]][%[[I]]] : memref<5xf32>
# CHECK-NEXT:    %[[R:.+]] = arith.maximumf %[[X]], %[[Y]] : f32
# CHECK-NEXT:    memref.store %[[R]], %out[%[[I]]] : memref<5xf32>

# Broadcasts are copied whatever reads them, here a matmul.
v, w = Var("v", (3,)), Var("w", (3, 4))
emit("broadcast_matmul", [v, w], broadcast_to(v, (2, 3)) @ w)
# CHECK-LABEL: func.func @broadcast_matmul(
# CHECK:       %[[T:.+]] = memref.alloc() : memref<2x3xf32>
# CHECK-NEXT:  scf.for %[[I:[a-z0-9_]+]] = %c0 to %c2 step %c1 {
# CHECK-NEXT:    scf.for %[[J:[a-z0-9_]+]] = %c0 to %c3 step %c1 {
# CHECK-NEXT:      %[[X:.+]] = memref.load %v[%[[J]]] : memref<3xf32>
# CHECK-NEXT:      memref.store %[[X]], %[[T]][%[[I]], %[[J]]] : memref<2x3xf32>
# CHECK:         memref.load %[[T]][%{{.+}}, %{{.+}}] : memref<2x3xf32>
