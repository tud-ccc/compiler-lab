# RUN: python %s | filecheck %s
# LoopEmitter: constants (NumPy arrays and numbers) become read-only globals.

import numpy as np

from algeco.frontend import Const, Var
from algeco.frontend.emit_loops import LoopEmitter

w = np.arange(6, dtype=np.float32).reshape(2, 3)
x = Var("x", (4, 2))
print(
    LoopEmitter().emit_module("f", [x], x @ Const(w, name="w") + 1.0 + Const(w[0] * 0))
)

# One global per distinct constant: "cst_<name>" for named constants, else "cst".
# CHECK:       "memref.global"() <{sym_name = "cst_w", type = memref<2x3xf32>, initial_value = dense<{{\[}}[0.000000e+00, 1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00, 5.000000e+00]]> : tensor<2x3xf32>, sym_visibility = "private", constant, alignment = 64 : i64}>
# CHECK-NEXT:  "memref.global"() <{sym_name = "cst", type = memref<f32>, initial_value = dense<1.000000e+00> : tensor<f32>, {{.*}}constant
# CHECK-NEXT:  "memref.global"() <{sym_name = "cst_1", type = memref<3xf32>, initial_value = dense<0.000000e+00> : tensor<3xf32>, {{.*}}constant
# CHECK-NEXT:  func.func @f(%x: memref<4x2xf32>, %out: memref<4x3xf32>) {

# The function reads them at the top, before any loop, like the other constants.
# CHECK-NEXT:    %w = memref.get_global @cst_w : memref<2x3xf32>
# CHECK-NOT:     scf.for
# CHECK:         %cst = memref.get_global @cst : memref<f32>
# CHECK-NOT:     scf.for
# CHECK:         %cst_1 = memref.get_global @cst_1 : memref<3xf32>
# CHECK:         memref.load %w[%{{.*}}, %{{.*}}] : memref<2x3xf32>
# A scalar is a 0-d buffer, loaded without indices.
# CHECK:         memref.load %cst[] : memref<f32>
# CHECK:         memref.load %cst_1[%{{.*}}] : memref<3xf32>
# CHECK-NOT:     memref.store %{{.*}}, %w[
