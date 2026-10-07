// RUN: algeco %s | filecheck %s

func.func @relu(%x: memref<4xf32>, %out: memref<4xf32>) {
  %zero = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %v = memref.load %x[%i] : memref<4xf32>
    %r = arith.maximumf %v, %zero : f32
    memref.store %r, %out[%i] : memref<4xf32>
  }
  func.return
}

// CHECK:      func.func @relu
// CHECK:        scf.for %{{.*}} = %c0 to %c4 step %c1 {
// CHECK-NEXT:     %{{.*}} = memref.load %x[%{{.*}}] : memref<4xf32>
// CHECK-NEXT:     %{{.*}} = arith.maximumf
