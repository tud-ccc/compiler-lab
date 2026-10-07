# RUN: python %s | filecheck %s
# LoopEmitter: matmul = an init nest, then a perfect i, j, k nest accumulating in memory.

from algeco.frontend import Var
from algeco.frontend.emit_loops import LoopEmitter

x, w = Var("x", (2, 3)), Var("w", (3, 4))
print(LoopEmitter().emit_module("matmul", [x, w], x @ w))

# The root node writes into the caller's buffer: no allocation.
# CHECK-LABEL: func.func @matmul(%x: memref<2x3xf32>, %w: memref<3x4xf32>, %out: memref<2x4xf32>) {
# CHECK-NOT:   memref.alloc

# Init: out[i, j] = 0 over (2, 4).
# CHECK:       scf.for %[[I:[a-z0-9_]+]] = %c0 to %c2 step %c1 {
# CHECK-NEXT:    scf.for %[[J:[a-z0-9_]+]] = %c0 to %c4 step %c1 {
# CHECK-NEXT:      memref.store %c0_f32, %out[%[[I]], %[[J]]] : memref<2x4xf32>
# CHECK-NEXT:    }
# CHECK-NEXT:  }

# Compute: out[i, j] += x[i, k] * w[k, j], loops in the order i, j, k.
# CHECK-NEXT:  scf.for %[[I:[a-z0-9_]+]] = %c0 to %c2 step %c1 {
# CHECK-NEXT:    scf.for %[[J:[a-z0-9_]+]] = %c0 to %c4 step %c1 {
# CHECK-NEXT:      scf.for %[[K:[a-z0-9_]+]] = %c0 to %c3 step %c1 {
# CHECK-NEXT:        %[[A:.+]] = memref.load %x[%[[I]], %[[K]]] : memref<2x3xf32>
# CHECK-NEXT:        %[[B:.+]] = memref.load %w[%[[K]], %[[J]]] : memref<3x4xf32>
# CHECK-NEXT:        %[[C:.+]] = memref.load %out[%[[I]], %[[J]]] : memref<2x4xf32>
# Strict floating point: no fastmath flags, so the sum is computed in loop order.
# CHECK-NEXT:        %[[P:.+]] = arith.mulf %[[A]], %[[B]] : f32
# CHECK-NEXT:        %[[S:.+]] = arith.addf %[[C]], %[[P]] : f32
# CHECK-NEXT:        memref.store %[[S]], %out[%[[I]], %[[J]]] : memref<2x4xf32>
# CHECK-NEXT:      }
# CHECK-NEXT:    }
# CHECK-NEXT:  }
# CHECK-NEXT:  func.return
# CHECK-NOT:   fastmath
