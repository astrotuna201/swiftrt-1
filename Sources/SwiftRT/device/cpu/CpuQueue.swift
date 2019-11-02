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

public final class CpuQueue: LocalDeviceQueue {
	// protocol properties
	public private(set) var trackingId = 0
    public var defaultQueueEventOptions = QueueEventOptions()
	public let device: ComputeDevice
    public let id: Int
	public let name: String
    public var logInfo: LogInfo
    public var timeout: TimeInterval?
    public var executeSynchronously: Bool = false
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()
    
    /// used to detect accidental queue access by other threads
    private let creatorThread: Thread
    /// the queue used for command execution
    private let commandQueue: DispatchQueue

    //--------------------------------------------------------------------------
    // initializers
    public init(logInfo: LogInfo, device: ComputeDevice, name: String, id: Int)
    {
        // create serial command queue
        commandQueue = DispatchQueue(label: "\(name).commandQueue")
        
        // create a completion event
        self.logInfo = logInfo
        self.device = device
        self.id = id
        self.name = name
        self.creatorThread = Thread.current
        let path = logInfo.namePath
        trackingId = ObjectTracker.global
            .register(self, namePath: path, isStatic: true)
        
        diagnostic("\(createString) DeviceQueue(\(trackingId)) " +
            "\(device.name)_\(name)", categories: .queueAlloc)
    }
    
    //--------------------------------------------------------------------------
    /// deinit
    /// waits for the queue to finish
    deinit {
        assert(Thread.current === creatorThread,
               "Queue has been captured and is being released by a " +
            "different thread. Probably by a queued function on the queue.")

        diagnostic("\(releaseString) DeviceQueue(\(trackingId)) " +
            "\(device.name)_\(name)", categories: [.queueAlloc])
        
        // release
        ObjectTracker.global.remove(trackingId: trackingId)

        // wait for the command queue to complete before shutting down
        do {
            try waitUntilQueueIsComplete()
        } catch {
            if let timeout = self.timeout {
                diagnostic("\(timeoutString) DeviceQueue(\(trackingId)) " +
                        "\(device.name)_\(name) timeout: \(timeout)",
                        categories: [.queueAlloc])
            }
        }
    }
    
    //--------------------------------------------------------------------------
    /// queues a closure on the queue for execution
    /// This will catch and propagate the last asynchronous error thrown.
    ///
    public func queue<Inputs, R>(
        _ functionName: @autoclosure () -> String,
        _ inputs: () throws -> Inputs,
        _ result: inout R,
        _ body: @escaping (Inputs, inout R.MutableValues) throws
        -> Void) where R: TensorView
    {
        // if the queue is in an error state, no additional work
        // will be queued
        guard lastError == nil else { return }
        
        // schedule the work
        diagnostic("\(schedulingString): \(functionName())",
            categories: .scheduling)
        
        do {
            // get the parameter sequences
            let input = try inputs()
            var sharedView = try result.sharedView(using: self)
            var results = try sharedView.mutableValues(using: self)
            
            if executeSynchronously {
                try body(input, &results)
            } else {
                // queue the work
                // report to device so we don't take a reference to `self`
                let errorDevice = device
                commandQueue.async {
                    do {
                        try body(input, &results)
                    } catch {
                        errorDevice.reportDevice(error: error)
                    }
                }
                diagnostic("\(schedulingString): \(functionName()) complete",
                    categories: .scheduling)
            }
        } catch {
            self.reportDevice(error: error)
        }
    }

    //--------------------------------------------------------------------------
    /// queues a closure on the queue for execution
    /// This will catch and propagate the last asynchronous error thrown.
    private func queue(_ body: @escaping () throws -> Void) {
        // if the queue is in an error state, no additional work
        // will be queued
        guard lastError == nil else { return }
        let errorDevice = device
        
        // make sure not to capture `self`
        func performBody() {
            do {
                try body()
            } catch {
                errorDevice.reportDevice(error: error)
            }
        }
        
        // queue the work
        if executeSynchronously {
            performBody()
        } else {
            commandQueue.async { performBody() }
        }
    }
    //--------------------------------------------------------------------------
    /// createEvent
    /// creates an event object used for queue synchronization
    public func createEvent(options: QueueEventOptions) throws -> QueueEvent {
        let event = CpuQueueEvent(options: options, timeout: timeout)
        diagnostic("\(createString) QueueEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .queueAlloc)
        return event
    }
    
    //--------------------------------------------------------------------------
    /// record(event:
    @discardableResult
    public func record(event: QueueEvent) throws -> QueueEvent {
        guard lastError == nil else { throw lastError! }
        let event = event as! CpuQueueEvent
        diagnostic("\(recordString) QueueEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .queueSync)
        
        // set event time
        if defaultQueueEventOptions.contains(.timing) {
            event.recordedTime = Date()
        }
        
        queue {
            event.signal()
        }
        return event
    }

    //--------------------------------------------------------------------------
    /// wait(for event:
    /// waits until the event has occurred
    public func wait(for event: QueueEvent) throws {
        guard lastError == nil else { throw lastError! }
        guard !event.occurred else { return }
        diagnostic("\(waitString) QueueEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .queueSync)
        
        queue {
            try event.wait()
        }
    }

    //--------------------------------------------------------------------------
    /// waitUntilQueueIsComplete
    /// blocks the calling thread until the command queue is empty
    public func waitUntilQueueIsComplete() throws {
        let event = try record(event: createEvent())
        diagnostic("\(waitString) QueueEvent(\(event.trackingId)) " +
            "waiting for \(device.name)_\(name) to complete",
            categories: .queueSync)
        try event.wait()
        diagnostic("\(signaledString) QueueEvent(\(event.trackingId)) on " +
            "\(device.name)_\(name)", categories: .queueSync)
    }
    
    //--------------------------------------------------------------------------
    /// perform indexed copy from source view to result view
    public func copy<T>(from view: T, to result: inout T) where T : TensorView {
        queue(#function, { try view.values() }, &result) {
            $0.map(to: &$1) { $0 }
        }
    }

    //--------------------------------------------------------------------------
    /// copies from one device array to another
    public func copyAsync(to array: DeviceArray,
                          from otherArray: DeviceArray) throws {
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        assert(array.buffer.count == otherArray.buffer.count,
               "buffer sizes don't match")
        queue {
            array.buffer.copyMemory(
                from: UnsafeRawBufferPointer(otherArray.buffer))
        }
    }

    //--------------------------------------------------------------------------
    /// copies a host buffer to a device array
    public func copyAsync(to array: DeviceArray,
                          from hostBuffer: UnsafeRawBufferPointer) throws
    {
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        assert(hostBuffer.baseAddress != nil)
        assert(array.buffer.count == hostBuffer.count,
               "buffer sizes don't match")
        queue {
            array.buffer.copyMemory(from: hostBuffer)
        }
    }
    
    //--------------------------------------------------------------------------
    /// copies a device array to a host buffer
    public func copyAsync(to hostBuffer: UnsafeMutableRawBufferPointer,
                          from array: DeviceArray) throws
    {
        assert(hostBuffer.baseAddress != nil)
        assert(array.buffer.count == hostBuffer.count,
               "buffer sizes don't match")
        queue {
            hostBuffer.copyMemory(from: UnsafeRawBufferPointer(array.buffer))
        }
    }

    //--------------------------------------------------------------------------
    /// fills the device array with zeros
    public func zero(array: DeviceArray) throws {
        assert(!array.isReadOnly, "cannot mutate read only reference buffer")
        queue {
            array.buffer.initializeMemory(as: UInt8.self, repeating: 0)
        }
    }
    
    //--------------------------------------------------------------------------
    /// simulateWork(x:timePerElement:result:
    /// introduces a delay in the queue by sleeping a duration of
    /// x.shape.elementCount * timePerElement
    public func simulateWork<T>(x: T, timePerElement: TimeInterval,
                                result: inout T)
        where T: TensorView
    {
        let delay = TimeInterval(x.shape.elementCount) * timePerElement
        delayQueue(atLeast: delay)
    }

    //--------------------------------------------------------------------------
    /// delayQueue(atLeast:
    /// causes the queue to sleep for the specified interval for testing
    public func delayQueue(atLeast interval: TimeInterval) {
        assert(Thread.current === creatorThread, queueThreadViolationMessage)
        queue {
            Thread.sleep(forTimeInterval: interval)
        }
    }
    
    //--------------------------------------------------------------------------
    /// throwTestError
    /// used for unit testing
    public func throwTestError() {
        assert(Thread.current === creatorThread, queueThreadViolationMessage)
        queue {
            throw DeviceError.queueError(idPath: [], message: "testError")
        }
    }
}
