// RUN: algeco %s -p algeco-interchange | filecheck %s
// RUN: algeco %s -p 'algeco-interchange{order=2,0,1}' | filecheck %s --check-prefix=FORCED

func.func @matmul(%x: memref<2x3xf32>, %w: memref<3x4xf32>, %out: memref<2x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %c3 step %c1 {
        %a = memref.load %x[%i, %k] : memref<2x3xf32>
        %b = memref.load %w[%k, %j] : memref<3x4xf32>
        %c = memref.load %out[%i, %j] : memref<2x4xf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        memref.store %s, %out[%i, %j] : memref<2x4xf32>
      }
    }
  }
  func.return
}

// The cost model moves j innermost: i, k, j. Bounds move with their loop,
// and the body is unchanged.
// CHECK-LABEL: func.func @matmul
// CHECK:       scf.for %i = %c0 to %c2 step %c1 {
// CHECK-NEXT:    scf.for %k = %c0 to %c3 step %c1 {
// CHECK-NEXT:      scf.for %j = %c0 to %c4 step %c1 {
// CHECK-NEXT:        %a = memref.load %x[%i, %k] : memref<2x3xf32>
// CHECK-NEXT:        %b = memref.load %w[%k, %j] : memref<3x4xf32>
// CHECK-NEXT:        %c = memref.load %out[%i, %j] : memref<2x4xf32>
// CHECK-NEXT:        %p = arith.mulf %a, %b : f32
// CHECK-NEXT:        %s = arith.addf %c, %p : f32
// CHECK-NEXT:        memref.store %s, %out[%i, %j] : memref<2x4xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:  }

// A forced order: new position -> old loop, so order=2,0,1 is k, i, j.
// FORCED-LABEL: func.func @matmul
// FORCED:       scf.for %k = %c0 to %c3 step %c1 {
// FORCED-NEXT:    scf.for %i = %c0 to %c2 step %c1 {
// FORCED-NEXT:      scf.for %j = %c0 to %c4 step %c1 {
