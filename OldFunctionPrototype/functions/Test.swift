//******************************************************************************
// Created by Edward Connell on 5/2/19
// Copyright © 2019 Connell Research. All rights reserved.
//
import Foundation

//==============================================================================
/// simulateWork(x:timePerElement:result:
/// introduces a delay in the queue by sleeping a duration of
/// x.shape.elementCount * timePerElement

/// in place
/// - Parameter x: value tensor
/// - Parameter timePerElement: seconds per element to delay
/// - Parameter result: x copied with delay
@inlinable @inline(__always)
public func simulateWork<T>(_ x: T, timePerElement: TimeInterval,
                            result: inout T)
    where T: TensorView
{
    DeviceContext.currentQueue
        .simulateWork(x: x, timePerElement: timePerElement, result: &result)
}

/// returns new view
/// - Parameter x: value tensor
/// - Returns: a new tensor containing `x` copied with delay
@inlinable @inline(__always)
//@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
public func simulateWork<T>(_ x: T, timePerElement: TimeInterval) -> T
    where T: TensorView
{
    var result = x.createDense()
    simulateWork(x, timePerElement: timePerElement, result: &result)
    return result
}

public extension TensorView {
    /// returns new view
    /// - Returns: a new tensor containing `x` copied with delay
    @inlinable @inline(__always)
    //@differentiable(vjp: _vjpLog(_:) where T: TensorFlowFloatingPoint)
    func simulateWork(timePerElement: TimeInterval = 0.0000001) -> Self {
        var result = createDense()
        SwiftRT.simulateWork(self, timePerElement: timePerElement,
                            result: &result)
        return result
    }
}

//==============================================================================
/// delayQueue(atLeast interval:
/// introduces a delay in the queue by sleeping a duration of

/// in place
/// - Parameter timePerElement: seconds per element to delay
@inlinable @inline(__always)
public func delayQueue(atLeast interval: TimeInterval) {
    DeviceContext.currentQueue.delayQueue(atLeast: interval)
}
