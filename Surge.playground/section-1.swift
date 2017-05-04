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

let A = Matrix([[1, 1, 1], [1, -1, -1], [4, -1, -1]])
var C = Matrix([[3], [1], [5]])

A

"C=";
C

do {
    let B = try inv(A) * C

    "B=";
    B
}
catch let error {
    print(error)
}

do {
    let pivot = try solve(a: A, b: &C)
    
    "B=";
    C
}
catch let error {
    print(error)
}


// MARK: - FFT

func plot<T>(values: [T], title: String) {
    for value in values {
         _=value
    }
}

let count = 64
let frequency = 4.0
let amplitude = 3.0

let x = (0..<count).map{ 2.0 * Double.pi / Double(count) * Double($0) * frequency }

//plot(values: sin(x), title:"Sine Wave")
plot(values: fft(sin(x)), title:"FFT")
