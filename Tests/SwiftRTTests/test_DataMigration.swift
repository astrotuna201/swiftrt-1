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

class test_DataMigration: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_stressCopyOnWriteDevice", test_stressCopyOnWriteDevice),
        ("test_viewMutateOnWrite", test_viewMutateOnWrite),
        ("test_tensorDataMigration", test_tensorDataMigration),
        ("test_mutateOnDevice", test_mutateOnDevice),
        ("test_copyOnWriteDevice", test_copyOnWriteDevice),
        ("test_copyOnWriteCrossDevice", test_copyOnWriteCrossDevice),
        ("test_copyOnWrite", test_copyOnWrite),
        ("test_columnMajorDataView", test_columnMajorDataView),
    ]
	
    //--------------------------------------------------------------------------
    // test_stressCopyOnWriteDevice
    // stresses view mutation and async copies on device
    func test_stressCopyOnWriteDevice() {
        do {
//            Platform.log.level = .diagnostic
//            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
            
            let matrix = Matrix<Float>((3, 2), name: "matrix", with: 0..<6)
            let index = (1, 1)
            
            for i in 0..<500 {
                var matrix2 = matrix
                try matrix2.set(value: 7, at: index)
                
                let value = try matrix2.value(at: index)
                if value != 7.0 {
                    XCTFail("i: \(i)  value is: \(value)")
                    break
                }
            }
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
	// test_viewMutateOnWrite
	func test_viewMutateOnWrite() {
		do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

            // create a Matrix and give it an optional name for logging
            var m0 = Matrix<Float>((3, 4), name: "weights", with: 0..<12)
            
            let _ = try m0.readWrite()
            XCTAssert(!m0.lastAccessMutatedView)
            let _ = try m0.readOnly()
            XCTAssert(!m0.lastAccessMutatedView)
            let _ = try m0.readWrite()
            XCTAssert(!m0.lastAccessMutatedView)
            
            // copy the view
            var m1 = m0
            // rw access m0 should mutate m0
            let _ = try m0.readWrite()
            XCTAssert(m0.lastAccessMutatedView)
            // m1 should now be unique reference
            XCTAssert(m1.isUniqueReference())
            let _ = try m1.readOnly()
            XCTAssert(!m1.lastAccessMutatedView)

            // copy the view
            var m2 = m0
            let _ = try m2.readOnly()
            XCTAssert(!m2.lastAccessMutatedView)
            // rw request should cause copy of m0 data
            let _ = try m2.readWrite()
            XCTAssert(m2.lastAccessMutatedView)
            // m2 should now be unique reference
            XCTAssert(m2.isUniqueReference())
            
        } catch {
			XCTFail(String(describing: error))
		}
	}
	
    //==========================================================================
    // test_tensorDataMigration
    //
    // This test uses the default UMA cpu queue, combined with the
    // testCpu1 and testCpu2 device queues.
    // The purpose is to test data replication and synchronization in the
    // following combinations.
    //
    // `app` means app thread
    // `uma` means any device that shares memory with the app thread
    // `discreet` is any device that does not share memory with the app thread
    // `same service` means moving data within (cuda gpu:0 -> gpu:1)
    // `cross service` means moving data between services
    //                 (cuda gpu:1 -> cpu cpu:0)
    //
    func test_tensorDataMigration() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

            // create a named queue on two different discreet devices
            // cpu devices 1 and 2 are discreet memory versions for testing
            let queue1 = Platform.testCpu1.queues[0]
            let queue2 = Platform.testCpu2.queues[0]

            // create a tensor and validate migration
            var view = Volume<Float>((2, 3, 4), with: 0..<24)
            
            _ = try view.readOnly()
            XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)

            _ = try view.readOnly()
            XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)

            // this device is not UMA so it
            // ALLOC device array on cpu:1
            // COPY  cpu:0 --> cpu:1_q0
            _ = try view.readOnly(using: queue1)
            XCTAssert(view.tensorArray.lastAccessCopiedBuffer)

            // write access hasn't been taken, so this is still up to date
            _ = try view.readOnly()
            XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)

            // an up to date copy is already there, so won't copy
            _ = try view.readWrite(using: queue1)
            XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)

            // ALLOC device array on cpu:2
            // COPY  cpu:1 --> cpu:2_q0
            _ = try view.readOnly(using: queue2)
            XCTAssert(view.tensorArray.lastAccessCopiedBuffer)
            
            _ = try view.readOnly(using: queue1)
            XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)

            _ = try view.readOnly(using: queue2)
            XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)

            _ = try view.readWrite(using: queue1)
            XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)

            // the master is on cpu:1 so we need to update cpu:2's version
            // COPY cpu:1 --> cpu:2_q0
            _ = try view.readOnly(using: queue2)
            XCTAssert(view.tensorArray.lastAccessCopiedBuffer)
            
            _ = try view.readWrite(using: queue2)
            XCTAssert(!view.tensorArray.lastAccessCopiedBuffer)

            // the master is on cpu:2 so we need to update cpu:1's version
            // COPY cpu:2 --> cpu:1_q0
            _ = try view.readWrite(using: queue1)
            XCTAssert(view.tensorArray.lastAccessCopiedBuffer)
            
            // the master is on cpu:1 so we need to update cpu:2's version
            // COPY cpu:1 --> cpu:2_q0
            _ = try view.readWrite(using: queue2)
            XCTAssert(view.tensorArray.lastAccessCopiedBuffer)
            
            // accessing data without a queue causes transfer to the host
            // COPY cpu:2_q0 --> uma:cpu:0
            _ = try view.readOnly()
            XCTAssert(view.tensorArray.lastAccessCopiedBuffer)

        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_mutateOnDevice
    func test_mutateOnDevice() {
    }

    //--------------------------------------------------------------------------
    // test_copyOnWriteDevice
    func test_copyOnWriteDevice() {
    }

    //--------------------------------------------------------------------------
    // test_copyOnWriteCrossDevice
    func test_copyOnWriteCrossDevice() {
    }

    //--------------------------------------------------------------------------
    // test_copyOnWrite
    // NOTE: uses the default queue
    func test_copyOnWrite() {
    }

    //--------------------------------------------------------------------------
    // test_columnMajorDataView
    // NOTE: uses the default queue
    //   0, 1,
    //   2, 3,
    //   4, 5
    func test_columnMajorDataView() {
        do {
            let cmMatrix = Matrix<Int32>((3, 2), layout: .columnMajor,
                                         elements: [0, 2, 4, 1, 3, 5])
            
            let expected = [Int32](0..<6)
            let values = try cmMatrix.array()
            XCTAssert(values == expected, "values don't match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
}
