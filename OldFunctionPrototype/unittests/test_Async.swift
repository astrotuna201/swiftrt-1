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
import XCTest
import Foundation

@testable import SwiftRT

class test_Async: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_hostMultiWrite", test_hostMultiWrite),
//        ("test_defaultQueueOp", test_defaultQueueOp),
//        ("test_secondaryDiscreetMemoryQueue", test_secondaryDiscreetMemoryQueue),
//        ("test_threeQueueInterleave", test_threeQueueInterleave),
//        ("test_tensorReferenceBufferSync", test_tensorReferenceBufferSync),
//        ("test_temporaryQueueShutdown", test_temporaryQueueShutdown),
//        ("test_QueueEventWait", test_QueueEventWait),
//        ("test_perfCreateQueueEvent", test_perfCreateQueueEvent),
//        ("test_perfRecordQueueEvent", test_perfRecordQueueEvent),
    ]

    //==========================================================================
    // test_hostMultiWrite
    // accesses a tensor on the host by dividing the first dimension
    // into batches and concurrently executing a user closure for each batch
    func test_hostMultiWrite() {
        do {
            Platform.log.level = .diagnostic
            typealias Pixel = RGB<UInt8>
            typealias ImageSet = Volume<Pixel>
            let expected = Pixel(0, 127, 255)
            var trainingSet = ImageSet((100, 256, 256))

            try trainingSet.hostMultiWrite { batch in
                for i in 0..<batch.items {
                    // get a view of the item at `i`
                    var itemView = batch.view(item: i)
                    
                    // get a writable buffer for the view
                    let buffer = itemView.getHostMultiWriteBuffer()
                    
                    // at this point load image data from a file or database,
                    // decompress, type convert, whatever is needed
                    // In this example we'll just fill the buffer with
                    // the `expected` value
                    buffer.initialize(repeating: expected)
                }
            }

            // check the last item to see if it contains the expected value
            let item = trainingSet.view(item: trainingSet.items - 1)
            let values = try item.array()
            XCTAssert(values[0] == expected)
        } catch {
            XCTFail(String(describing: error))
        }
        
        // check for object leaks
        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_defaultQueueOp
    // initializes two matrices and adds them together
    func test_defaultQueueOp() {
        do {
            Platform.log.level = .diagnostic

            let m1 = Matrix<Int32>((2, 5), name: "m1", with: 0..<10)
            let m2 = Matrix<Int32>((2, 5), name: "m2", with: 0..<10)
            let result = m1 + m2
            let values = try result.array()
            
            let expected = (0..<10).map { Int32($0 * 2) }
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_secondaryDiscreetMemoryQueue
    // initializes two matrices on the app thread, executes them on `queue1`,
    // the retrieves the results
    func test_secondaryDiscreetMemoryQueue() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .scheduling, .queueSync]

            let device1 = Platform.testCpu1
            
            let m1 = Matrix<Int32>((2, 5), name: "m1", with: 0..<10)
            let m2 = Matrix<Int32>((2, 5), name: "m2", with: 0..<10)

            // perform on user provided discreet memory queue
            let result = using(device1) { m1 + m2 }

            // synchronize with host queue and retrieve result values
            let values = try result.array()
            
            let expected = (0..<10).map { Int32($0 * 2) }
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_threeQueueInterleave
    func test_threeQueueInterleave() {
        do {
            Platform.log.level = .diagnostic
            
            let device1 = Platform.testCpu1
            let device2 = Platform.testCpu2

            let m1 = Matrix<Int32>((2, 3), name: "m1", with: 0..<6)
            let m2 = Matrix<Int32>((2, 3), name: "m2", with: 0..<6)
            let m3 = Matrix<Int32>((2, 3), name: "m3", with: 0..<6)

            // sum the values with a delay on device 1
            let sum_m1m2: Matrix<Int32> = using(device1) {
                delayQueue(atLeast: 0.1)
                return m1 + m2
            }

            // multiply the values on device 2
            let result = using(device2) {
                sum_m1m2 * m3
            }

            // synchronize with host queue and retrieve result values
            let values = try result.array()
            
            let expected = (0..<6).map { Int32(($0 + $0) * $0) }
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_tensorReferenceBufferSync
    func test_tensorReferenceBufferSync() {
    }

    //==========================================================================
    // test_QueueEventWait
    func test_QueueEventWait() {
        do {
            Platform.log.level = .diagnostic
            Platform.local.log.categories = [.queueSync]
            
            let queue = Platform.testCpu1.queues[0]
            let event = try queue.createEvent()
            queue.delayQueue(atLeast: 0.001)
            try queue.record(event: event).wait()
            XCTAssert(event.occurred, "wait failed to block")
        } catch {
            XCTFail(String(describing: error))
        }

        if ObjectTracker.global.hasUnreleasedObjects {
            XCTFail(ObjectTracker.global.getActiveObjectReport())
        }
    }
    
    //==========================================================================
    // test_perfCreateQueueEvent
    // measures the event overhead of creating 10,000 events
    func test_perfCreateQueueEvent() {
        #if !DEBUG
        do {
            let queue = try Platform.local.createQueue()
            self.measure {
                do {
                    for _ in 0..<10000 {
                        _ = try queue.createEvent()
                    }
                } catch {
                    XCTFail(String(describing: error))
                }
            }
        } catch {
            XCTFail(String(describing: error))
        }
        #endif
    }

    //==========================================================================
    // test_perfRecordQueueEvent
    // measures the event overhead of processing 10,000 tensors
    func test_perfRecordQueueEvent() {
        #if !DEBUG
        do {
            let queue = Platform.testCpu1.queues[0]
            self.measure {
                do {
                    for _ in 0..<10000 {
                        _ = try queue.record(event: queue.createEvent())
                    }
                    try queue.waitUntilQueueIsComplete()
                } catch {
                    XCTFail(String(describing: error))
                }
            }
        } catch {
            XCTFail(String(describing: error))
        }
        #endif
    }
}
