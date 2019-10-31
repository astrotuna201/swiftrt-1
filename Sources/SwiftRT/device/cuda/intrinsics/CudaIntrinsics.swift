//******************************************************************************
// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
import Foundation

#if canImport(CCuda)

public extension CudaQueue {
    //--------------------------------------------------------------------------
    /// abs
    func abs<T>(x: T, result: inout T) where
    T: TensorView, T.Element: FloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// add
    func add<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element: Numeric
    {
    }

    //--------------------------------------------------------------------------
    /// all
    func all<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
    T: TensorView, T.Element == Bool
    {
    }

    //--------------------------------------------------------------------------
    /// any
    func any<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
    T: TensorView, T.Element == Bool
    {
    }

    //--------------------------------------------------------------------------
    /// approximatelyEqual
    func approximatelyEqual<T>(lhs: T, rhs: T,
                               tolerance: T.Element,
                               result: inout T.BoolView) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// argmax
    func argmax<T>(x: T, along axes: Vector<IndexElement>?,
                   result: inout T.IndexView) where
    T: TensorView, T.Element: Numeric
    {

    }

    //--------------------------------------------------------------------------
    /// argmin
    func argmin<T>(x: T, along axes: Vector<IndexElement>?,
                   result: inout T.IndexView) where
    T: TensorView, T.Element: Numeric
    {

    }

    //--------------------------------------------------------------------------
    /// asum
    func asum<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
    T: TensorView, T.Element: FloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// cast
    func cast<T, R>(from: T, to result: inout R) where
    T: TensorView, R: TensorView, T.Element: AnyConvertable,
    R.Element : AnyConvertable
    {

    }

    //--------------------------------------------------------------------------
    /// ceil
    func ceil<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// concatenate
    func concatenate<T>(view: T, with other: T,
                        alongAxis axis: Int, result: inout T) where
    T: TensorView
    {

    }

    //--------------------------------------------------------------------------
    /// copies the elements from view to result
    func copy<T>(view: T, result: inout T) where T: TensorView
    {
    }

    //--------------------------------------------------------------------------
    /// cos
    func cos<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// cosh
    func cosh<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// div
    func div<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element: FloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// equal
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Element: Equatable
    {
    }

    //--------------------------------------------------------------------------
    /// exp
    func exp<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// fill(result:with:
    /// NOTE: this can be much faster, doesn't need to be ordered access
    func fill<T>(_ result: inout T, with value: T.Element) where
        T: TensorView
    {
    }

    //--------------------------------------------------------------------------
    /// fillWithIndex(x:startAt:
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
    T: TensorView, T.Element: AnyNumeric
    {
    }

    //--------------------------------------------------------------------------
    /// floor
    func floor<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// greater
    func greater<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Element: Comparable
    {
    }

    //--------------------------------------------------------------------------
    /// greaterOrEqual
    func greaterOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Element: Comparable
    {
    }

    //--------------------------------------------------------------------------
    /// less
    func less<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Element: Comparable
    {
    }

    //--------------------------------------------------------------------------
    /// lessOrEqual
    func lessOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Element: Comparable
    {
    }

    //--------------------------------------------------------------------------
    /// log(x:result:
    func log<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// logicalNot(x:result:
    func logicalNot<T>(x: T, result: inout T) where
    T: TensorView, T.Element == Bool
    {
    }

    //--------------------------------------------------------------------------
    /// logicalAnd(x:result:
    func logicalAnd<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element == Bool
    {
    }

    //--------------------------------------------------------------------------
    /// logicalOr(x:result:
    func logicalOr<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element == Bool
    {
    }

    //--------------------------------------------------------------------------
    /// logSoftmax(x:result:
    func logSoftmax<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// matmul(lhs:rhs:result:
    func matmul<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element: Numeric
    {

    }

    //--------------------------------------------------------------------------
    /// max
    func max<T>(x: T, along axes: [Int], result: inout T) where
    T: TensorView, T.Element: Numeric
    {

    }

    //--------------------------------------------------------------------------
    /// maximum(lhs:rhs:result:
    func maximum<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element: Comparable
    {
    }

    //--------------------------------------------------------------------------
    /// mean
    func mean<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
    T: TensorView, T.Element: FloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    /// min
    func min<T>(x: T, along axes: [Int], result: inout T) where
    T: TensorView, T.Element: Numeric
    {

    }

    //--------------------------------------------------------------------------
    /// minimum(lhs:rhs:result:
    func minimum<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element: Comparable
    {
    }

    //--------------------------------------------------------------------------
    // mod
    func mod<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    // mul
    func mul<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element: Numeric
    {
    }

    //--------------------------------------------------------------------------
    // neg
    func neg<T>(x: T, result: inout T) where
    T: TensorView, T.Element: SignedNumeric
    {
    }

    //--------------------------------------------------------------------------
    // notEqual
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Element: Equatable
    {
    }

    //--------------------------------------------------------------------------
    // pow
    func pow<T>(x: T, y: T, result: inout T) where
    T: TensorView, T.Element: AnyNumeric
    {
    }

    //--------------------------------------------------------------------------
    // prod
    func prod<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
    T: TensorView, T.Element: AnyNumeric
    {
    }

    //--------------------------------------------------------------------------
    // rsqrt
    func rsqrt<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    // replacing
    func replacing<T>(x: T, with other: T, where mask: T.BoolView,
                      result: inout T) where T: TensorView
    {
    }

    //--------------------------------------------------------------------------
    // sin
    func sin<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    // sinh
    func sinh<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    // square
    func square<T>(x: T, result: inout T) where
    T: TensorView, T.Element: FloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    // squaredDifference
    func squaredDifference<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element: FloatingPoint
    {
        assert(lhs.shape.elementCount == rhs.shape.elementCount,
               "tensors must have equal element counts")
    }

    //--------------------------------------------------------------------------
    // sqrt
    func sqrt<T>(x: T, result: inout T) where
    T: TensorView, T.Element: FloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    // subtract
    func subtract<T>(lhs: T, rhs: T, result: inout T) where
    T: TensorView, T.Element: Numeric
    {
    }

    //--------------------------------------------------------------------------
    // sum
    func sum<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
    T: TensorView, T.Element: Numeric
    {
    }

    //--------------------------------------------------------------------------
    // tan
    func tan<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }

    //--------------------------------------------------------------------------
    // tanh
    func tanh<T>(x: T, result: inout T) where
    T: TensorView, T.Element: AnyFloatingPoint
    {
    }
}

#endif
