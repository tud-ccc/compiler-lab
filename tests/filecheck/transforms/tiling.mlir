// RUN: algeco %s -p 'algeco-strip-mine{sizes=1,2,3}' | filecheck %s
// RUN: algeco %s -p 'algeco-tile{sizes=2,2,3}' | filecheck %s --check-prefix=TILE

func.func @matmul(%x: memref<4x3xf32>, %w: memref<3x4xf32>, %out: memref<4x4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %c3 step %c1 {
        %a = memref.load %x[%i, %k] : memref<4x3xf32>
        %b = memref.load %w[%k, %j] : memref<3x4xf32>
        %c = memref.load %out[%i, %j] : memref<4x4xf32>
        %p = arith.mulf %a, %b : f32
        %s = arith.addf %c, %p : f32
        memref.store %s, %out[%i, %j] : memref<4x4xf32>
      }
    }
  }
  func.return
}

// Only j is strip-mined: 1 is no tile, and 3 is the whole k loop. The tile
// size is a constant before the nest, and %j is computed in the innermost body.
// CHECK-LABEL: func.func @matmul
// CHECK:       %[[T:c2.*]] = arith.constant 2 : index
// CHECK-NEXT:  scf.for %i = %c0 to %c4 step %c1 {
// CHECK-NEXT:    scf.for %jj = %c0 to %c4 step %[[T]] {
// CHECK-NEXT:      scf.for %j0 = %c0 to %[[T]] step %c1 {
// CHECK-NEXT:        scf.for %k = %c0 to %c3 step %c1 {
// CHECK-NEXT:          %j = arith.addi %jj, %j0 : index
// CHECK-NEXT:          %a = memref.load %x[%i, %k] : memref<4x3xf32>
// CHECK-NEXT:          %b = memref.load %w[%k, %j] : memref<3x4xf32>
// CHECK-NEXT:          %c = memref.load %out[%i, %j] : memref<4x4xf32>

// Tiling: the tile loops of i and j move out, k stays a point loop.
// TILE-LABEL: func.func @matmul
// TILE:       scf.for %ii = %c0 to %c4 step %[[TI:c2.*]] {
// TILE-NEXT:    scf.for %jj = %c0 to %c4 step %[[TJ:c2.*]] {
// TILE-NEXT:      scf.for %i0 = %c0 to %[[TI]] step %c1 {
// TILE-NEXT:        scf.for %j0 = %c0 to %[[TJ]] step %c1 {
// TILE-NEXT:          scf.for %k = %c0 to %c3 step %c1 {
// TILE-NEXT:            %j = arith.addi %jj, %j0 : index
// TILE-NEXT:            %i = arith.addi %ii, %i0 : index
// TILE-NEXT:            %a = memref.load %x[%i, %k] : memref<4x3xf32>
