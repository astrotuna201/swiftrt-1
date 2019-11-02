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

//==============================================================================
// QueueIntrinsics
/// The required set of base level intrinsic functions for a `DeviceQueue`
public protocol DeviceIntrinsics {
    /// Computes the absolute value of the specified TensorView element-wise.
    func abs<T>(x: T, result: inout T) where
        T: TensorView, T.Element: FloatingPoint
    /// Adds two tensors and produces their sum.
    func add<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: Numeric
    /// Returns `true` if all values are `true`. Otherwise, returns `false`.
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    func all<T>(x: T, along axes: Vector<IndexElement>?, result: inout T)
        where T: TensorView, T.Element == Bool
    /// Returns `true` if any values are`true`. Otherwise, returns `false`.
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    func any<T>(x: T, along axes: Vector<IndexElement>?, result: inout T)
        where T: TensorView, T.Element == Bool
    /// Performs a point wise comparison within the specified tolerance
    func approximatelyEqual<T>(lhs: T, rhs: T,
                               tolerance: T.Element,
                               result: inout T.BoolView) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// Returns the indices of the maximum values along the specified axes. The
    /// reduced dimensions are removed.
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`
    func argmax<T>(x: T, along axes: Vector<IndexElement>?,
                   result: inout T.IndexView) where
        T: TensorView, T.Element: Numeric
    /// Returns the indices of the minimum values along the specified axes. The
    /// reduced dimensions are removed.
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`
    func argmin<T>(x: T, along axes: Vector<IndexElement>?,
                   result: inout T.IndexView) where
        T: TensorView, T.Element: Numeric
    /// Sums the absolute value of the input along the specified axes
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    func asum<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Element: FloatingPoint
    /// cast scalar types
    /// - Parameter from: the input data
    /// - Parameter result: the output
    func cast<T, R>(from: T, to result: inout R) where
        T: TensorView, T.Element: AnyConvertable,
        R: TensorView, R.Element: AnyConvertable
    /// Computes the ceiling of the specified TensorView element-wise.
    func ceil<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// Concatenates tensors along the specified axis.
    /// - Precondition: The tensors must have the same dimensions, except for the
    ///                 specified axis.
    /// - Precondition: The axis must be in the range `-rank..<rank`.
    func concatenate<T>(view: T, with other: T, alongAxis axis: Int,
                        result: inout T) where T: TensorView
    /// copies the elements from view to result
    func copy<T>(view: T, result: inout T) where T: TensorView
    /// Computes the element-wise `cos`
    func cos<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// Computes the element-wise `cosh`
    func cosh<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// Returns the quotient of dividing the first TensorView by the second.
    /// - Note: `/` supports broadcasting.
    func div<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: FloatingPoint
    /// Computes `lhs == rhs` element-wise and returns a `TensorView` of Boolean
    /// values.
    /// - Note: `.==` supports broadcasting.
    func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
        T: TensorView, T.Element: Equatable
    /// Computes the element-wise `exp`
    func exp<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// fills the view with the scalar value
    func fill<T>(_ result: inout T, with: T.Element) where T: TensorView
    /// fills the view with the spatial sequential index
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T: TensorView, T.Element: AnyNumeric
    /// Computes the element-wise `floor`
    func floor<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// Computes `lhs > rhs` element-wise and returns a tensor of Bool values.
    func greater<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable
    /// Computes `lhs >= rhs` element-wise and returns a tensor of Bool values.
    func greaterOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable
    /// Computes `lhs < rhs` element-wise and returns a tensor of Bool values.
    func less<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable
    /// Computes `lhs <= rhs` element-wise and returns a tensor of Bool values.
    func lessOrEqual<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Comparable
    /// Computes the element-wise `log`
    func log<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// Computes the element-wise `!x`
    func logicalNot<T>(x: T, result: inout T) where
        T: TensorView, T.Element == Bool
    /// Computes the element-wise `lhs && rhs`
    func logicalAnd<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element == Bool
    /// Computes the element-wise `lhs || rhs`
    func logicalOr<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element == Bool
    /// Computes the element-wise `logSoftmax`
    func logSoftmax<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// Performs matrix multiplication with another TensorView and produces the
    /// result.
    func matmul<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Numeric
    /// Returns the maximum values along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    func max<T>(x: T, along axes: [Int], result: inout T) where
        T: TensorView, T.Element: Numeric
    /// Computes the element-wise maximum of two tensors.
    /// - Note: `max` supports broadcasting.
    func maximum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    /// Returns the arithmetic mean along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`
    func mean<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Element: FloatingPoint
    /// Returns the minimum values along the specified axes. The reduced
    /// dimensions are removed.
    /// - Parameter axes: The dimensions to reduce.
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`
    func min<T>(x: T, along axes: [Int], result: inout T) where
        T: TensorView, T.Element: Numeric
    /// Computes the element-wise minimum of two tensors.
    /// - Note: `max` supports broadcasting.
    func minimum<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: Comparable
    /// Returns the remainder of dividing the first TensorView by the second.
    /// - Note: `%` supports broadcasting.
    func mod<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: AnyFloatingPoint
    /// mul
    func mul<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: Numeric
    /// Computes the element-wise negation
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Element: SignedNumeric
    /// Computes `lhs != rhs` element-wise and returns a tensor of Bools
    /// - Note: `.==` supports broadcasting.
    func notEqual<T>(lhs: T, rhs: T, result: inout T.BoolView)
        where T: TensorView, T.Element: Numeric
    /// Computes the element-wise `x**y`
    func pow<T>(x: T, y: T, result: inout T)
        where T: TensorView, T.Element: AnyNumeric
    /// Product of the input elements to produce a scalar
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    /// - Precondition: Each value in `axes` must be in the range `-rank...rank`
    func prod<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Element: AnyNumeric
    /// Replaces elements of `x` with `other` in the lanes where `mask` is`true`
    ///
    /// - Precondition: `x` and `other` must have the same shape. If
    ///   `x` and `other` are scalar, then `mask` must also be scalar. If
    ///   `x` and `other` have rank greater than or equal to `1`, then `mask`
    ///   must be either have the same shape as `self` or be a 1-D `TensorView`
    ///   such that `mask.scalarCount == self.shape[0]`.
    func replacing<T>(x: T, with other: T, where mask: T.BoolView,
                      result: inout T) where T: TensorView
    /// Computes the element-wise `rsqrt`
    func rsqrt<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// Computes the element-wise `sin`
    func sin<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// Computes the element-wise `sinh`
    func sinh<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// Computes the element-wise `square`
    func square<T>(x: T, result: inout T) where
        T: TensorView, T.Element: FloatingPoint
    /// Computes the element-wise `(lhs - rhs)**2`
    func squaredDifference<T>(lhs: T, rhs: T, result: inout T) where
        T: TensorView, T.Element: FloatingPoint
    /// Computes the element-wise `sqrt`
    func sqrt<T>(x: T, result: inout T) where
        T: TensorView, T.Element: FloatingPoint
    /// subtract
    func subtract<T>(lhs: T, rhs: T, result: inout T)
        where T: TensorView, T.Element: Numeric
    /// Sums the input along the specified axes
    /// - Parameter x: the tensor value
    /// - Parameter axes: The axes to reduce
    func sum<T>(x: T, along axes: Vector<IndexElement>?, result: inout T) where
        T: TensorView, T.Element: Numeric
    /// Computes the element-wise `tan`
    func tan<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
    /// Computes the element-wise `tanh`
    func tanh<T>(x: T, result: inout T) where
        T: TensorView, T.Element: AnyFloatingPoint
}
