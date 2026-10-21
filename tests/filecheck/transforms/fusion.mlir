// RUN: algeco %s -p algeco-fuse-loops | filecheck %s

// The broadcast of a bias row, then the add: fused into one nest, the bias
// forwarded into the add, and the copy %t removed with its alloc and dealloc.
func.func @bias(%x: memref<2x4xf32>, %b: memref<4xf32>, %out: memref<2x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %t = memref.alloc() : memref<2x4xf32>
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %v = memref.load %b[%j] : memref<4xf32>
      memref.store %v, %t[%i, %j] : memref<2x4xf32>
    }
  }
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %a = memref.load %x[%i, %j] : memref<2x4xf32>
      %w = memref.load %t[%i, %j] : memref<2x4xf32>
      %s = arith.addf %a, %w : f32
      memref.store %s, %out[%i, %j] : memref<2x4xf32>
    }
  }
  memref.dealloc %t : memref<2x4xf32>
  func.return
}

// CHECK-LABEL: func.func @bias
// CHECK-NOT:   memref.alloc
// CHECK:       scf.for %i = %c0 to %c2 step %c1 {
// CHECK-NEXT:    scf.for %j = %c0 to %c4 step %c1 {
// CHECK-NEXT:      %v = memref.load %b[%j] : memref<4xf32>
// CHECK-NEXT:      %a = memref.load %x[%i, %j] : memref<2x4xf32>
// CHECK-NEXT:      %s = arith.addf %a, %v : f32
// CHECK-NEXT:      memref.store %s, %out[%i, %j] : memref<2x4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NOT:   memref.dealloc
// CHECK:       func.return

// A matmul (i, j, k) and the add of its result (i, j): fused at depth 2. The
// accumulator %t stays: its store is inside the k loop, so nothing is forwarded.
func.func @matmul_add(%x: memref<2x3xf32>, %w: memref<3x4xf32>, %b: memref<2x4xf32>, %out: memref<2x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %t = memref.alloc() : memref<2x4xf32>
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %c3 step %c1 {
        %a = memref.load %x[%i, %k] : memref<2x3xf32>
        %v = memref.load %w[%k, %j] : memref<3x4xf32>
        %c = memref.load %t[%i, %j] : memref<2x4xf32>
        %p = arith.mulf %a, %v : f32
        %s = arith.addf %c, %p : f32
        memref.store %s, %t[%i, %j] : memref<2x4xf32>
      }
    }
  }
  scf.for %i = %c0 to %c2 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      %acc = memref.load %t[%i, %j] : memref<2x4xf32>
      %bias = memref.load %b[%i, %j] : memref<2x4xf32>
      %r = arith.addf %acc, %bias : f32
      memref.store %r, %out[%i, %j] : memref<2x4xf32>
    }
  }
  memref.dealloc %t : memref<2x4xf32>
  func.return
}

// CHECK-LABEL: func.func @matmul_add
// CHECK:       %t = memref.alloc() : memref<2x4xf32>
// CHECK-NEXT:  scf.for %i = %c0 to %c2 step %c1 {
// CHECK-NEXT:    scf.for %j = %c0 to %c4 step %c1 {
// CHECK-NEXT:      scf.for %k = %c0 to %c3 step %c1 {
// CHECK:           }
// CHECK-NEXT:      %acc = memref.load %t[%i, %j] : memref<2x4xf32>
// CHECK-NEXT:      %bias = memref.load %b[%i, %j] : memref<2x4xf32>
// CHECK-NEXT:      %r = arith.addf %acc, %bias : f32
// CHECK-NEXT:      memref.store %r, %out[%i, %j] : memref<2x4xf32>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
// CHECK-NEXT:  memref.dealloc %t : memref<2x4xf32>
