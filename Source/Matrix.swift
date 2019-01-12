// Hyperbolic.swift
//
// Copyright (c) 2014–2015 Mattt Thompson (http://mattt.me)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

import Accelerate
public typealias LAInt = __CLPK_integer // = Int32

enum MatrixError: Error {
    case matrixIsNotInvertable
    case equationsIsUnsolved(Int)
    case equationsArgIsIllegal(Int)
}

public enum MatrixAxies {
    case row
    case column
}

public class Matrix<T> where T: FloatingPoint, T: ExpressibleByFloatLiteral {
    public typealias MatrixElement = T
    
    public let rows: Int
    public let columns: Int
    public var grid: [MatrixElement]
    
    public init(rows: Int, columns: Int, repeatedValue: MatrixElement) {
        self.rows = rows
        self.columns = columns
        
        self.grid = [MatrixElement](repeating: repeatedValue, count: rows * columns)
    }
    
    public convenience init(_ contents: [[MatrixElement]]) {
        let m: Int = contents.count
        let n: Int = contents[0].count
        let repeatedValue: MatrixElement = 0.0
        
        self.init(rows: m, columns: n, repeatedValue: repeatedValue)
        
        for (i, row) in contents.enumerated() {
            grid.replaceSubrange(i*n..<i*n+Swift.min(m, row.count), with: row)
        }
    }
    
    public subscript(row: Int, column: Int) -> MatrixElement {
        get {
            assert(indexIsValidForRow(row, column: column))
            return grid[(row * columns) + column]
        }
        
        set {
            assert(indexIsValidForRow(row, column: column))
            grid[(row * columns) + column] = newValue
        }
    }
    
    public subscript(row row: Int) -> [MatrixElement] {
        get {
            assert(row < rows)
            let startIndex = row * columns
            let endIndex = row * columns + columns
            return Array(grid[startIndex..<endIndex])
        }
        
        set {
            assert(row < rows)
            assert(newValue.count == columns)
            let startIndex = row * columns
            let endIndex = row * columns + columns
            grid.replaceSubrange(startIndex..<endIndex, with: newValue)
        }
    }
    
    public subscript(column column: Int) -> [MatrixElement] {
        get {
            var result = [MatrixElement](repeating: 0.0, count: rows)
            for i in 0..<rows {
                let index = i * columns + column
                result[i] = self.grid[index]
            }
            return result
        }
        
        set {
            assert(column < columns)
            assert(newValue.count == rows)
            for i in 0..<rows {
                let index = i * columns + column
                grid[index] = newValue[i]
            }
        }
    }
    
    fileprivate func indexIsValidForRow(_ row: Int, column: Int) -> Bool {
        return row >= 0 && row < rows && column >= 0 && column < columns
    }
}

// MARK: - Printable

extension Matrix: CustomStringConvertible {
    public var description: String {
        var description = ""
        
        for i in 0..<rows {
            let contents = (0..<columns).map{"\(self[i, $0])"}.joined(separator: "\t")
            
            switch (i, rows) {
            case (0, 1):
                description += "(\t\(contents)\t)"
            case (0, _):
                description += "⎛\t\(contents)\t⎞"
            case (rows - 1, _):
                description += "⎝\t\(contents)\t⎠"
            default:
                description += "⎜\t\(contents)\t⎥"
            }
            
            description += "\n"
        }
        
        return description
    }
}

// MARK: - SequenceType

extension Matrix: Sequence{
    public func makeIterator() -> AnyIterator<ArraySlice<MatrixElement>> {
        let endIndex = rows * columns
        var nextRowStartIndex = 0
        
        return AnyIterator {
            if nextRowStartIndex == endIndex {
                return nil
            }
            
            let currentRowStartIndex = nextRowStartIndex
            nextRowStartIndex += self.columns
            
            return self.grid[currentRowStartIndex..<nextRowStartIndex]
        }
    }
}

extension Matrix: Equatable {}
public func ==<T> (lhs: Matrix<T>, rhs: Matrix<T>) -> Bool {
    return lhs.rows == rhs.rows && lhs.columns == rhs.columns && lhs.grid == rhs.grid
}


// MARK: -

public func add(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")
    
    let results = y
    cblas_saxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)
    
    return results
}

public func add(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix dimensions not compatible with addition")
    
    let results = y
    cblas_daxpy(Int32(x.grid.count), 1.0, x.grid, 1, &(results.grid), 1)
    
    return results
}

public func mul(_ alpha: Float, x: Matrix<Float>) -> Matrix<Float> {
    let results = x
    cblas_sscal(Int32(x.grid.count), alpha, &(results.grid), 1)
    
    return results
}

public func mul(_ alpha: Double, x: Matrix<Double>) -> Matrix<Double> {
    let results = x
    cblas_dscal(Int32(x.grid.count), alpha, &(results.grid), 1)
    
    return results
}

public func mul(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.columns == y.rows, "Matrix dimensions not compatible with multiplication")
    
    let results = Matrix<Float>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, &(results.grid), Int32(results.columns))
    
    return results
}

public func mul(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.columns == y.rows, "Matrix dimensions not compatible with multiplication")
    
    let results = Matrix<Double>(rows: x.rows, columns: y.columns, repeatedValue: 0.0)
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(x.rows), Int32(y.columns), Int32(x.columns), 1.0, x.grid, Int32(x.columns), y.grid, Int32(y.columns), 0.0, &(results.grid), Int32(results.columns))
    
    return results
}

public func elmul(_ x: Matrix<Double>, y: Matrix<Double>) -> Matrix<Double> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix must have the same dimensions")
    let result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = x.grid * y.grid
    return result
}

public func elmul(_ x: Matrix<Float>, y: Matrix<Float>) -> Matrix<Float> {
    precondition(x.rows == y.rows && x.columns == y.columns, "Matrix must have the same dimensions")
    let result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = x.grid * y.grid
    return result
}

public func div(_ x: Matrix<Double>, y: Matrix<Double>) throws -> Matrix<Double> {
    let yInv = try inv(y)
    precondition(x.columns == yInv.rows, "Matrix dimensions not compatible")
    return mul(x, y: yInv)
}

public func div(_ x: Matrix<Float>, y: Matrix<Float>) throws -> Matrix<Float> {
    let yInv = try inv(y)
    precondition(x.columns == yInv.rows, "Matrix dimensions not compatible")
    return mul(x, y: yInv)
}

public func pow(_ x: Matrix<Double>, _ y: Double) -> Matrix<Double> {
    let result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = pow(x.grid, y)
    return result
}

public func pow(_ x: Matrix<Float>, _ y: Float) -> Matrix<Float> {
    let result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = pow(x.grid, y)
    return result
}

public func exp(_ x: Matrix<Double>) -> Matrix<Double> {
    let result = Matrix<Double>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = exp(x.grid)
    return result
}

public func exp(_ x: Matrix<Float>) -> Matrix<Float> {
    let result = Matrix<Float>(rows: x.rows, columns: x.columns, repeatedValue: 0.0)
    result.grid = exp(x.grid)
    return result
}

public func sum(_ x: Matrix<Double>, axies: MatrixAxies = .column) -> Matrix<Double> {
    
    switch axies {
    case .column:
        let result = Matrix<Double>(rows: 1, columns: x.columns, repeatedValue: 0.0)
        for i in 0..<x.columns {
            result.grid[i] = sum(x[column: i])
        }
        return result
        
    case .row:
        let result = Matrix<Double>(rows: x.rows, columns: 1, repeatedValue: 0.0)
        for i in 0..<x.rows {
            result.grid[i] = sum(x[row: i])
        }
        return result
    }
}

public func inv(_ x : Matrix<Float>) throws -> Matrix<Float> {
    precondition(x.rows == x.columns, "Matrix must be square")
    
    let results = x
    
    var ipiv = [__CLPK_integer](repeating: 0, count: x.rows * x.rows)
    var lwork = __CLPK_integer(x.columns * x.columns)
    var work = [CFloat](repeating: 0.0, count: Int(lwork))
    var error: __CLPK_integer = 0
    var nc = __CLPK_integer(x.columns)
    var ncc = nc
    var nccc = nc
    sgetrf_(&nc, &nccc, &(results.grid), &ncc, &ipiv, &error)
    sgetri_(&nc, &(results.grid), &ncc, &ipiv, &work, &lwork, &error)
    
    guard error == 0 else {
        throw MatrixError.matrixIsNotInvertable
    }
    
    return results
}

public func inv(_ x : Matrix<Double>) throws-> Matrix<Double> {
    precondition(x.rows == x.columns, "Matrix must be square")
    
    let results = x
    
    var ipiv = [__CLPK_integer](repeating: 0, count: x.rows * x.rows)
    var lwork = __CLPK_integer(x.columns * x.columns)
    var work = [CDouble](repeating: 0.0, count: Int(lwork))
    var error: __CLPK_integer = 0
    var nc = __CLPK_integer(x.columns)
    var ncc = nc
    var nccc = nc
    
    dgetrf_(&nc, &nccc, &(results.grid), &ncc, &ipiv, &error)
    dgetri_(&nc, &(results.grid), &ncc, &ipiv, &work, &lwork, &error)
    
    guard error == 0 else {
        throw MatrixError.matrixIsNotInvertable
    }
    
    return results
}

public func transpose(_ x: Matrix<Float>) -> Matrix<Float> {
    let results = Matrix<Float>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    vDSP_mtrans(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))
    
    return results
}

public func transpose(_ x: Matrix<Double>) -> Matrix<Double> {
    let results = Matrix<Double>(rows: x.columns, columns: x.rows, repeatedValue: 0.0)
    vDSP_mtransD(x.grid, 1, &(results.grid), 1, vDSP_Length(results.rows), vDSP_Length(results.columns))
    
    return results
}

// MARK: - Operators

public func + (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return add(lhs, y: rhs)
}

public func + (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return add(lhs, y: rhs)
}

public func * (lhs: Float, rhs: Matrix<Float>) -> Matrix<Float> {
    return mul(lhs, x: rhs)
}

public func * (lhs: Double, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, x: rhs)
}

public func * (lhs: Matrix<Float>, rhs: Matrix<Float>) -> Matrix<Float> {
    return mul(lhs, y: rhs)
}

public func * (lhs: Matrix<Double>, rhs: Matrix<Double>) -> Matrix<Double> {
    return mul(lhs, y: rhs)
}

public func / (lhs: Matrix<Double>, rhs: Matrix<Double>) throws -> Matrix<Double> {
    return try div(lhs, y: rhs)
}

public func / (lhs: Matrix<Float>, rhs: Matrix<Float>) throws -> Matrix<Float> {
    return try div(lhs, y: rhs)
}

public func / (lhs: Matrix<Double>, rhs: Double) -> Matrix<Double> {
    let result = Matrix<Double>(rows: lhs.rows, columns: lhs.columns, repeatedValue: 0.0)
    result.grid = lhs.grid / rhs;
    return result;
}

public func / (lhs: Matrix<Float>, rhs: Float) -> Matrix<Float> {
    let result = Matrix<Float>(rows: lhs.rows, columns: lhs.columns, repeatedValue: 0.0)
    result.grid = lhs.grid / rhs;
    return result;
}

postfix operator ′
public postfix func ′ (value: Matrix<Float>) -> Matrix<Float> {
    return transpose(value)
}

public postfix func ′ (value: Matrix<Double>) -> Matrix<Double> {
    return transpose(value)
}

@discardableResult public func solve(a A:Matrix<Float>, b B: inout Matrix<Float>) throws -> [LAInt] {
    let outB =  Matrix<Float>(rows:B.rows, columns:B.columns, repeatedValue:0)
    var pivot = [LAInt]()
    if B.columns >= 1 {
        
        var v = Matrix<Float>(rows:B.rows,
                              columns:1,
                              repeatedValue:0)
        
        for c in 0..<B.columns {
            v[column:0] = B[column:c]

            pivot.append(contentsOf: try _solve(a: A, b: &v))
            
            outB[column:c] = v[column:0]
        }
        
        B = outB
    }
    else {
        throw MatrixError.equationsArgIsIllegal(Int(-1))
    }
    return pivot
}

@discardableResult public func solve(a A:Matrix<Double>, b B: inout Matrix<Double>) throws -> [LAInt] {
    let outB =  Matrix<Double>(rows:B.rows, columns:B.columns, repeatedValue:0)
    var pivot = [LAInt]()
    if B.columns >= 1 {
        
        var v = Matrix<Double>(rows:B.rows,
                              columns:1,
                              repeatedValue:0)
        
        for c in 0..<B.columns {
            v[column:0] = B[column:c]
            let p = try _solve(a: A, b: &v)
            pivot.append(contentsOf: p)
            outB[column:c] = v[column:0]
        }
        
        B = outB
    }
    else {
        throw MatrixError.equationsArgIsIllegal(Int(-1))
    }
    return pivot
}

@discardableResult private func _solve(a A:Matrix<Float>, b B: inout Matrix<Float>) throws -> [LAInt] {
    
    let A  = transpose(A)
    
    let equations = A.rows
    
    var numberOfEquations = LAInt(A.rows)
    var columnsInA        = LAInt(A.columns)
    var elementsInB       = LAInt(B.rows)
    var bSolutionCount    = LAInt(1)
    
    var outputOk: LAInt = 0
    var pivot = [LAInt](repeating: 0, count: equations)
    
    if B.columns == 1 {
        
        var grid = [Float](A.grid)
        for c in 0..<B.columns {
            
            var v = B[column:c]
            sgesv_( &numberOfEquations, &bSolutionCount, &grid, &columnsInA, &pivot, &v, &elementsInB, &outputOk)
            
            guard outputOk == 0 else {
                if outputOk < 0 {
                    throw MatrixError.equationsArgIsIllegal(Int(-outputOk))
                }
                else {
                    throw MatrixError.equationsIsUnsolved(Int(outputOk))
                }
            }
            
            B[column:c] = v
        }
    }
    else {
        throw MatrixError.equationsArgIsIllegal(Int(-1))
    }
    return pivot
}

@discardableResult private func _solve(a A:Matrix<Double>, b B: inout Matrix<Double>) throws -> [LAInt] {
    
    let A  = transpose(A)
    
    let equations = A.rows
    
    var numberOfEquations = LAInt(A.rows)
    var columnsInA        = LAInt(A.columns)
    var elementsInB       = LAInt(B.rows)
    var bSolutionCount    = LAInt(1)
    
    var outputOk: LAInt = 0
    var pivot = [LAInt](repeating: 0, count: equations)
    
    if B.columns == 1 {
        
        var grid = [Double](A.grid)
        for c in 0..<B.columns {
            
            var v = B[column:c]
            dgesv_( &numberOfEquations, &bSolutionCount, &grid, &columnsInA, &pivot, &v, &elementsInB, &outputOk)
            
            guard outputOk == 0 else {
                if outputOk < 0 {
                    throw MatrixError.equationsArgIsIllegal(Int(-outputOk))
                }
                else {
                    throw MatrixError.equationsIsUnsolved(Int(outputOk))
                }
            }
            
            B[column:c] = v
        }
    }
    else {
        throw MatrixError.equationsArgIsIllegal(Int(-1))
    }
    return pivot
}
