//******************************************************************************
// Created by Edward Connell on 4/3/19
// Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// all(x:alongAxes:)
/// Returns `true` if all values are equal to `true` along the specified
/// axes. Otherwise returns `false`. The result extent along the specified
/// axes will be 1. Rank is not reduced.

/// in place
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func all<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil,
                   result: inout T)
    where T: TensorView, T.Element == Bool
{
    DeviceContext.currentQueue.all(x: x, along: axes, result: &result)
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element == Bool {
    @inlinable @inline(__always)
    func all(alongAxes: Int...) -> Self {
        // turn into a vector
        let axes = Vector<IndexElement>(
            with: shape.makePositive(indices: alongAxes))
        var result = createDense()
        SwiftRT.all(self, alongAxes: axes, result: &result)
        return result
    }
    
    @inlinable @inline(__always)
    func all() -> Self {
        let extents = [Int](repeating: 1, count: shape.rank)
        var result = createDense(with: extents)
        SwiftRT.all(self, result: &result)
        return result
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func all(squeezing: Int...) -> NDTensor<Element> {
        let axes = shape.makePositive(indices: squeezing)
        let axesVec = Vector<IndexElement>(with: axes)
        var result = createDense()
        SwiftRT.all(self, alongAxes: axesVec, result: &result)
        return result.squeezed(axes: axes)
    }
}

//==============================================================================
/// mean(x:alongAxes:)
/// Returns the mean of all values along the specified
/// axes. The result extent along the specified axes will be 1.
/// Rank is not reduced.

/// to result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func mean<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil,
                    result: inout T)
    where T: TensorView, T.Element: BinaryFloatingPoint
{
    DeviceContext.currentQueue.mean(x: x, along: axes, result: &result)
}

/// return result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func mean<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil) -> T
    where T: TensorView, T.Element: BinaryFloatingPoint
{
    let extents = [Int](repeating: 1, count: x.rank)
    var result = x.createDense(with: extents)
    DeviceContext.currentQueue.mean(x: x, along: axes, result: &result)
    return result
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element: BinaryFloatingPoint {
    @inlinable @inline(__always)
    func mean(alongAxes axes: Vector<IndexElement>? = nil) -> Self {
        var result = createDense()
        SwiftRT.mean(self, alongAxes: axes, result: &result)
        return result
    }

    @inlinable @inline(__always)
    func mean(alongAxes: [Int]) -> Self {
        // turn into a vector
        let axes = Vector<IndexElement>(
            with: shape.makePositive(indices: alongAxes))
        return mean(alongAxes: axes)
    }

    @inlinable @inline(__always)
    func mean(alongAxes: Int...) -> Self {
        return mean(alongAxes: alongAxes)
    }
    
    @inlinable @inline(__always)
    func mean() -> Self {
        let extents = [Int](repeating: 1, count: shape.rank)
        var result = createDense(with: extents)
        SwiftRT.mean(self, result: &result)
        return result
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func mean(squeezingAxes: Int...) -> NDTensor<Element> {
        let axes = shape.makePositive(indices: squeezingAxes)
        let axesVec = Vector<IndexElement>(with: axes)
        var result = createDense()
        SwiftRT.mean(self, alongAxes: axesVec, result: &result)
        return result.squeezed(axes: axes)
    }
}

//==============================================================================
/// sum(x:alongAxes:)
/// Returns the sum of all values are along the specified
/// axes. The result extent along the specified axes will be 1.
/// Rank is not reduced.

/// to result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func sum<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil,
                   result: inout T)
    where T: TensorView, T.Element: Numeric
{
    DeviceContext.currentQueue.sum(x: x, along: axes, result: &result)
}

/// return result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func sum<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil) -> T
    where T: TensorView, T.Element: Numeric
{
    let extents = [Int](repeating: 1, count: x.rank)
    var result = x.createDense(with: extents)
    DeviceContext.currentQueue.sum(x: x, along: axes, result: &result)
    return result
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element: Numeric {
    @inlinable @inline(__always)
    func sum(alongAxes: Int...) -> Self {
        // turn into a vector
        let axes = Vector<IndexElement>(
            with: shape.makePositive(indices: alongAxes))
        var result = createDense()
        SwiftRT.sum(self, alongAxes: axes, result: &result)
        return result
    }
    
    @inlinable @inline(__always)
    func sum() -> Self {
        let extents = [Int](repeating: 1, count: shape.rank)
        var result = createDense(with: extents)
        SwiftRT.sum(self, result: &result)
        return result
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func sum(squeezingAxes: Int...) -> NDTensor<Element> {
        let axes = shape.makePositive(indices: squeezingAxes)
        let axesVec = Vector<IndexElement>(with: axes)
        var result = createDense()
        SwiftRT.sum(self, alongAxes: axesVec, result: &result)
        return result.squeezed(axes: axes)
    }
}

//==============================================================================
/// variance(x:alongAxes:)
/// Returns the variance of all values are along the specified
/// axes. The result extent along the specified axes will be 1.
/// Rank is not reduced.

/// to result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter meanValue: the tensor where the mean will be written
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func variance<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil,
                        mean: inout T, result: inout T)
    where T: TensorView, T.Element: BinaryFloatingPoint
{
//    if let axes = axes {
//    } else {
    DeviceContext.currentQueue.mean(x: x, along: axes, result: &mean)
    let meanVec = T(with: x.extents, repeating: mean)
    DeviceContext.currentQueue.mean(x: squaredDifference(x, meanVec),
                          along: axes, result: &result)
//    }
}

/// return result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func variance<T>(_ x: T, alongAxes axes: Vector<IndexElement>? = nil)
    -> (mean: T, variance: T)
    where T: TensorView, T.Element: BinaryFloatingPoint
{
    var meanX = x.createDense(with: [1])
    var result = meanX
    SwiftRT.variance(x, alongAxes: axes, mean: &meanX, result: &result)
    return (meanX, result)
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element: BinaryFloatingPoint {
    @inlinable @inline(__always)
    func variance(alongAxes: [Int]) -> (mean: Self, variance: Self) {
        // turn into a vector
        let positiveAxes = shape.makePositive(indices: alongAxes)
        let axes = Vector<IndexElement>(with: positiveAxes)
        return SwiftRT.variance(self, alongAxes: axes)
    }

    @inlinable @inline(__always)
    func variance(alongAxes: Int...) -> (mean: Self, variance: Self) {
        return variance(alongAxes: alongAxes)
    }
    
    @inlinable @inline(__always)
    func variance() -> (mean: Self, variance: Self) {
        return SwiftRT.variance(self)
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func variance(squeezingAxes: Int...) ->
        (mean: NDTensor<Element>, variance: NDTensor<Element>)
    {
        let axes = shape.makePositive(indices: squeezingAxes)
        let varianceVal = variance(alongAxes: squeezingAxes)
        return (varianceVal.mean.squeezed(axes: axes),
                varianceVal.variance.squeezed(axes: axes))
    }
}

//==============================================================================
/// standardDeviation(x:alongAxes:)
/// Returns the variance of all values are along the specified
/// axes. The result extent along the specified axes will be 1.
/// Rank is not reduced.

/// to result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter meanValue: the tensor where the mean will be written
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func standardDeviation<T>(_ x: T,
                                 alongAxes axes: Vector<IndexElement>? = nil,
                                 mean: inout T, result: inout T)
    where T: TensorView, T.Element: BinaryFloatingPoint
{
    var _variance = x.createDense(with: [1])
    SwiftRT.variance(x, alongAxes: axes, mean: &mean, result: &_variance)
    DeviceContext.currentQueue.sqrt(x: _variance, result: &result)
}

/// return result
/// - Parameter x: value tensor
/// - Parameter alongAxes: the axes to operate on
/// - Parameter result: the scalar tensor where the result will be written
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
@inlinable @inline(__always)
public func standardDeviation<T>(_ x: T,
                                 alongAxes axes: Vector<IndexElement>? = nil)
    -> (mean: T, variance: T)
    where T: TensorView, T.Element: BinaryFloatingPoint
{
    var meanX = x.createDense(with: [1])
    var result = meanX
    SwiftRT.standardDeviation(x, alongAxes: axes, mean: &meanX, result: &result)
    return (meanX, result)
}

/// returns new view
/// - Parameter alongAxes: the axes to operate on
/// - Returns: a new tensor containing the result
/// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
public extension TensorView where Element: BinaryFloatingPoint {
    @inlinable @inline(__always)
    func standardDeviation(alongAxes: [Int]) -> (mean: Self, variance: Self) {
        // turn into a vector
        let positiveAxes = shape.makePositive(indices: alongAxes)
        let axes = Vector<IndexElement>(with: positiveAxes)
        return SwiftRT.standardDeviation(self, alongAxes: axes)
    }
    
    @inlinable @inline(__always)
    func standardDeviation(alongAxes: Int...) -> (mean: Self, variance: Self) {
        return variance(alongAxes: alongAxes)
    }
    
    @inlinable @inline(__always)
    func standardDeviation() -> (mean: Self, variance: Self) {
        return SwiftRT.standardDeviation(self)
    }
    
    /// returns new view
    /// - Parameter alongAxes: the axes to operate on
    /// - Returns: a new NDTensor containing the result
    /// - Precondition: Each value in `axes` must be in the range `-rank..<rank`.
    @inlinable @inline(__always)
    func standardDeviation(squeezingAxes: Int...) ->
        (mean: NDTensor<Element>, variance: NDTensor<Element>)
    {
        let axes = shape.makePositive(indices: squeezingAxes)
        let varianceVal = variance(alongAxes: squeezingAxes)
        return (varianceVal.mean.squeezed(axes: axes),
                varianceVal.variance.squeezed(axes: axes))
    }
}
