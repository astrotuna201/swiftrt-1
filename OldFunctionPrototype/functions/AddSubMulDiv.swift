//******************************************************************************
// Created by Edward Connell on 4/3/19
// Copyright © 2019 Connell Research. All rights reserved.
//
// Inspired by the Google S4TF project
//
/// TensorView operators are defined in several forms
/// - in place: the result is written to a tensor provided by the caller
///
/// - return new view: a new result tensor is created and returned. This is
///   less efficient in iterative cases, but convenient for expression
///   composition.
///
/// - operator form: + - * / etc..
///
/// - scalar arg form: one of the arguments might be passed as a scalar and
///   converted to a scalar tensor for the caller as a convenience.
///
/// - scalar type mismatch: a form is provided to allow the user to pass an
///   integer value where a Float or Double is needed.
///   For example:
///     let m = Matrix<Float>()
///     let x = m + 1
import Foundation

infix operator ++  : AdditionPrecedence
infix operator .<  : ComparisonPrecedence
infix operator .<= : ComparisonPrecedence
infix operator .>= : ComparisonPrecedence
infix operator .>  : ComparisonPrecedence
infix operator .== : ComparisonPrecedence
infix operator .!= : ComparisonPrecedence
infix operator .=

//==============================================================================
/// Add tensors
/// Adds two tensors to produce their sum

/// in place
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the extents are smaller than `lhs`
///   then broadcasting will be performed via repeated indexing.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpAdd(lhs:rhs:) where Element : TensorFlowFloatingPoint)
public func add<T>(_ lhs: T, _ rhs: T, result: inout T)
    where T: TensorView, T.Element: Numeric
{
    DeviceContext.currentQueue.add(lhs: lhs, rhs: rhs, result: &result)
}

/// returns new view
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the extents are smaller than
///   `lhs` then broadcasting is performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func add<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: Numeric
{
    var result = lhs.createDense()
    add(lhs, rhs, result: &result)
    return result
}

public extension TensorView where Element: Numeric {
    /// operator
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func + (lhs: Self, rhs: Self) -> Self {
        return add(lhs, rhs)
    }
}

public extension TensorView where Element: FloatingPoint & AnyFloatingPoint {
    /// operator (Self + element)
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand scalar. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func +(lhs: Self, rhs: Element) -> Self {
        return add(lhs, lhs.create(value: rhs))
    }
}

public extension TensorView where Element: BinaryInteger {
    /// operator (Self + scalar)
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand scalar. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func +(lhs: Self, rhs: Element) -> Self {
        return add(lhs, lhs.create(value: rhs))
    }
}

//==============================================================================
/// Subtract tensors
/// Subtracts (left - right) with broadcasting

/// in place
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the size is smaller than `lhs` then
///   broadcasting will be performed via repeated indexing.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//    @differentiable(vjp: _vjpSubtract(lhs:rhs:) where Element: TensorFlowFloatingPoint)
public func subtract<T>(_ lhs: T, _ rhs: T, result: inout T)
    where T: TensorView, T.Element: Numeric
{
    DeviceContext.currentQueue.subtract(lhs: lhs, rhs: rhs, result: &result)
}

/// returning new view
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the extents are smaller than
///   `lhs` then broadcasting is performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func subtract<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: Numeric
{
    var result = lhs.createDense()
    subtract(lhs, rhs, result: &result)
    return result
}

public extension TensorView where Element: Numeric {
    /// operator
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func - (lhs: Self, rhs: Self) -> Self {
        return subtract(lhs, rhs)
    }
}

public extension TensorView where Element: FloatingPoint {
    /// operator (Self - scalar)
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand scalar. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func - (lhs: Self, rhs: Element) -> Self {
        return subtract(lhs, lhs.create(value: rhs))
    }
}

public extension TensorView where Element: BinaryInteger {
    /// operator (Self - scalar)
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand scalar. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func - (lhs: Self, rhs: Element) -> Self {
        return subtract(lhs, lhs.create(value: rhs))
    }
}

//==============================================================================
/// Element wise multiply tensors with broadcasting

/// in place
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the size is smaller than `lhs` then
///   broadcasting will be performed via repeated indexing.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpMultiply(lhs:rhs:) where Element : TensorFlowFloatingPoint)
public func mul<T>(_ lhs: T, _ rhs: T, result: inout T)
    where T: TensorView, T.Element: Numeric
{
    DeviceContext.currentQueue.mul(lhs: lhs, rhs: rhs, result: &result)
}

/// returning new view
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the extents are smaller than
///   `lhs` then broadcasting is performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func mul<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: Numeric
{
    var result = lhs.createDense()
    mul(lhs, rhs, result: &result)
    return result
}

public extension TensorView where Element: Numeric {
    /// operator
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func * (lhs: Self, rhs: Self) -> Self {
        return mul(lhs, rhs)
    }
}

public extension TensorView where Element: FloatingPoint {
    /// operator (Self - scalar)
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand scalar. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func * (lhs: Self, rhs: Element) -> Self {
        return mul(lhs, lhs.create(value: rhs))
    }
}

public extension TensorView where Element: BinaryInteger {
    /// operator (Self - scalar)
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand scalar. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func * (lhs: Self, rhs: Element) -> Self {
        return mul(lhs, lhs.create(value: rhs))
    }
}

//==============================================================================
/// Element wise divide tensors with broadcasting

/// in place
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the size is smaller than `lhs` then
///   broadcasting will be performed via repeated indexing.
/// - Parameter result: the tensor where the result will be written
@inlinable @inline(__always)
//@differentiable(vjp: _vjpDivide(lhs:rhs:) where Element : TensorFlowFloatingPoint)
public func div<T>(_ lhs: T, _ rhs: T, result: inout T)
    where T: TensorView, T.Element: FloatingPoint
{
    DeviceContext.currentQueue.div(lhs: lhs, rhs: rhs, result: &result)
}

/// returning new view
/// - Parameter lhs: left hand tensor
/// - Parameter rhs: right hand tensor. If the extents are smaller than
///   `lhs` then broadcasting is performed via repeated indexing.
/// - Returns: a new tensor containing the result
@inlinable @inline(__always)
public func div<T>(_ lhs: T, _ rhs: T) -> T
    where T: TensorView, T.Element: FloatingPoint
{
    var result = lhs.createDense()
    div(lhs, rhs, result: &result)
    return result
}

public extension TensorView where Element: FloatingPoint {
    /// operator
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand tensor. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func / (lhs: Self, rhs: Self) -> Self {
        return div(lhs, rhs)
    }
}

public extension TensorView where Element: FloatingPoint {
    /// operator (Self - scalar)
    /// - Parameter lhs: left hand tensor
    /// - Parameter rhs: right hand scalar. If the extents are smaller than
    ///   `lhs` then broadcasting is performed via repeated indexing.
    /// - Returns: a new tensor containing the result
    @inlinable @inline(__always)
    static func / (lhs: Self, rhs: Element) -> Self {
        let scalarTensor = lhs.create(value: rhs)
        return div(lhs, scalarTensor)
    }
}
