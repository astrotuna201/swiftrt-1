//******************************************************************************
// Created by Edward Connell on 4/3/19
// Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// copy
/// copies the elements from `view` to `result`

/// with placement
/// - Parameter view: tensor to be copied
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
public func copy<T>(view: T, result: inout T) where T: TensorView
{
    DeviceContext.currentQueue.copy(view: view, result: &result)
}

//==============================================================================
/// abs(x)
/// computes the absolute value of `x`

/// with placement
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func abs<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    DeviceContext.currentQueue.abs(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func abs<T>(_ x: T) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    var result = x.createDense()
    abs(x, result: &result)
    return result
}

public extension TensorView where Element: FloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpAbs(_:) where T: TensorFlowFloatingPoint)
    func abs() -> Self {
        var result = createDense()
        SwiftRT.abs(self, result: &result)
        return result
    }
}

//==============================================================================
/// log(x)
/// computes the log of `x`

/// with placement
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func log<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: AnyFloatingPoint
{
    DeviceContext.currentQueue.log(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func log<T>(_ x: T) -> T
    where T: TensorView, T.Element: AnyFloatingPoint
{
    var result = x.createDense()
    log(x, result: &result)
    return result
}

public extension TensorView where Element: AnyFloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
    func log() -> Self {
        var result = createDense()
        SwiftRT.log(self, result: &result)
        return result
    }
}

//==============================================================================
/// logSoftmax(x)
/// computes the logSoftmax of `x`

/// with placement
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLogSoftmax(_:) where T: TensorFlowFloatingPoint)
public func logSoftmax<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: AnyFloatingPoint
{
    DeviceContext.currentQueue.logSoftmax(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLogSoftmax(_:) where T: TensorFlowFloatingPoint)
public func logSoftmax<T>(_ x: T) -> T
    where T: TensorView, T.Element: AnyFloatingPoint
{
    var result = x.createDense()
    logSoftmax(x, result: &result)
    return result
}

public extension TensorView where Element: AnyFloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpLogSoftmax(_:) where T: TensorFlowFloatingPoint)
    func logSoftmax() throws -> Self {
        var result = createDense()
        SwiftRT.logSoftmax(self, result: &result)
        return result
    }
}

//==============================================================================
/// pow(x, y)
/// raises tensor 'x' to the tensor 'y' power

/// with placement
/// - Parameter x: value tensor
/// - Parameter y: exponent tensor. If the extents are smaller than `x` then
///   broadcasting will be performed via repeated indexing.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpPow(_:_:) where T: TensorFlowFloatingPoint)
public func pow<T>(_ x: T, _ y: T, result: inout T)
    where T: TensorView, T.Element: AnyNumeric
{
    DeviceContext.currentQueue.pow(x: x, y: y, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Parameter y: exponent tensor. If the extents are smaller than `x` then
///   broadcasting will be performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpPow(_:_:) where T: TensorFlowFloatingPoint)
public func pow<T>(_ x: T, _ y: T) -> T
    where T: TensorView, T.Element: AnyNumeric
{
    var result = x.createDense()
    pow(x, y, result: &result)
    return result
}

public extension TensorView where Element: AnyNumeric {
    /// returns new view
    /// - Parameter y: exponent tensor. If the extents are smaller than `x` then
    ///   broadcasting will be performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpPow(_:_:) where T: TensorFlowFloatingPoint)
    func pow(_ y: Self) -> Self{
        var result = createDense()
        SwiftRT.pow(self, y, result: &result)
        return result
    }
}
public extension TensorView where Element: AnyNumeric {
    /// returns new view
    /// - Parameter y: exponent tensor. If the extents are smaller than `x` then
    ///   broadcasting will be performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpPow(_:_:) where T: TensorFlowFloatingPoint)
    func pow(_ y: Element) -> Self {
        var result = createDense()
        SwiftRT.pow(self, create(value: y), result: &result)
        return result
    }
}

//==============================================================================
/// fill<T>(result:value:
/// fills the view with the specified value
public func fill<T>(_ result: inout T, with value: T.Element) where
    T: TensorView
{
    DeviceContext.currentQueue.fill(&result, with: value)
}

public extension TensorView {
    func filled(with value: Element) -> Self {
        var result = createDense()
        DeviceContext.currentQueue.fill(&result, with: value)
        return result
    }
}

//==============================================================================
/// fillWithIndex(x:startAt:
/// fills the view with the spatial sequential index
public func fillWithIndex<T>(_ result: inout T, startAt index: Int = 0) where
    T: TensorView, T.Element: AnyNumeric
{
    DeviceContext.currentQueue.fillWithIndex(&result, startAt: index)
}

public extension TensorView where Element: AnyNumeric {
    func filledWithIndex(startAt index: Int = 0) -> Self {
        var result = createDense()
        DeviceContext.currentQueue.fillWithIndex(&result, startAt: index)
        return result
    }
}

//==============================================================================
/// Computes `lhs == rhs` element-wise and returns a `TensorView` of Boolean
/// values.
public func equal<T>(lhs: T, rhs: T, result: inout T.BoolView) where
    T: TensorView, T.Element: Equatable
{
    DeviceContext.currentQueue.equal(lhs: lhs, rhs: rhs, result: &result)
}

public extension TensorView where Element: Equatable
{
    /// operator (Self - scalar)
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand scalar. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func .== (lhs: Self, rhs: Self) -> BoolView {
        var result = lhs.createBoolTensor()
        equal(lhs: lhs, rhs: rhs, result: &result)
        return result
    }
}

//==============================================================================
/// square(x)
/// computes the square value of `x`

/// with placement
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func square<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    DeviceContext.currentQueue.square(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func square<T>(_ x: T) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    var result = x.createDense()
    square(x, result: &result)
    return result
}

public extension TensorView where Element: FloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpAbs(_:) where T: TensorFlowFloatingPoint)
    func squared() -> Self {
        var result = createDense()
        SwiftRT.square(self, result: &result)
        return result
    }
}

//==============================================================================
/// squaredDifference tensors
/// (lhs - rhs)**2

/// in place
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpAdd(lhs:rhs:) where Element : TensorFlowFloatingPoint)
public func squaredDifference<T>(_ lhs: T, _ rhs: T, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    DeviceContext.currentQueue
        .squaredDifference(lhs: lhs, rhs: rhs, result: &result)
}

/// returns new view
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func squaredDifference<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    var result = lhs.createDense()
    squaredDifference(lhs, rhs, result: &result)
    return result
}

//==============================================================================
/// sqrt(x)
/// computes the element wise square root value of `x`

/// with placement
/// - Parameter x: value tensor
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func sqrt<T>(_ x: T, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    DeviceContext.currentQueue.sqrt(x: x, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func sqrt<T>(_ x: T) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    var result = x.createDense()
    sqrt(x, result: &result)
    return result
}

public extension TensorView where Element: FloatingPoint {
    /// returns new view
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpAbs(_:) where T: TensorFlowFloatingPoint)
    func sqrt() -> Self {
        var result = createDense()
        SwiftRT.sqrt(self, result: &result)
        return result
    }
}

