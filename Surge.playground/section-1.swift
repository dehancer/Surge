import Foundation
import Surge
//import XCPlayground
//import PlaygroundSupport

// MARK: - Arithmetic

let n = [-1.0, 2.0, 3.0, 4.0, 5.0]
let sum = Surge.sum(n)

let a = [1.0, 3.0, 5.0, 7.0]
let b = [2.0, 4.0, 6.0, 8.0]
let product = Surge.mul(a, y: b)

// MARK: - Matrix

// ⎛ 1  1 ⎞       ⎛ 3 ⎞
// ⎢      ⎟ * B = ⎢   ⎟         C = ?
// ⎝ 1 -1 ⎠       ⎝ 1 ⎠

var A = Matrix<Float>([
    [1,  1,  1],
    [1, -1, -1],
    [4, -1, -2]
    ])

var C  = Matrix<Float>([[3], [1], [5]])
var C1 = Matrix<Float>([[1], [2], [2]])
var C12 = Matrix<Float>([[1], [2], [3]])

var C2 = Matrix<Float>([[3,1,1],
                 [1,2,2],
                 [5,2,3]])


print("A=\n\(A)")
print("C=\n\(C)")
print("C1=\n\(C1)")
print("C2=\n\(C2)")

do {
    let B = try inv(A) * C

    print("solved as inversion C=\n\(B)")
}
catch let error {
    print(error)
}

do {
    let B = try inv(A) * C1
    
    print("solved as inversion C1=\n\(B)")
}
catch let error {
    print(error)
}


do {
    let pivot = try solve(a: A, b: &C)
    
    print("solved C=\n\(C)")
}
catch let error {
    print(error)
}

do {
    let pivot = try solve(a: A, b: &C1)
    
    print("solved C1=\n\(C1)")
}
catch let error {
    print(error)
}

do {
    let pivot = try solve(a: A, b: &C12)
    
    print("solved C12=\n\(C12)")
}
catch let error {
    print(error)
}

do {
    let pivot = try solve(a: A, b: &C2)
    
    print("solved C2=\n\(C2)")
}
catch let error {
    print(error)
}



// MARK: - FFT
//
//func plot<T>(values: [T], title: String) {
//    for value in values {
//         _=value
//    }
//}
//
//let count = 64
//let frequency = 4.0
//let amplitude = 3.0
//
//let x = (0..<count).map{ 2.0 * Double.pi / Double(count) * Double($0) * frequency }
//
////plot(values: sin(x), title:"Sine Wave")
//plot(values: fft(sin(x)), title:"FFT")
