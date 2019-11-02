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
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

            // create a named queue on two different discreet devices
            // cpu devices 1 and 2 are discreet memory versions for testing
            let device1 = Platform.testCpu1
            let queue1 = device1.queues[0]
            let device2 = Platform.testCpu2
            let queue2 = device2.queues[0]

            // create a Matrix on device 1 and fill with indexes
            // memory is only allocated on device 1. This also shows how a
            // temporary can be used in a scope. No memory is copied.
            var matrix = using(device1) {
                Matrix<Float>((3, 2)).filledWithIndex()
            }

            // retreive value on app thread
            // memory is allocated in the host app space and the data is copied
            // from device 1 to the host using queue 0.
            let value1 = try matrix.value(at: (1, 1))
            XCTAssert(value1 == 3.0)

            // simulate a readonly kernel access on device 1.
            // matrix was not previously modified, so it is up to date
            // and no data movement is necessary
            _ = try matrix.readOnly(using: queue1)

            // sum device 1 copy, which should equal 15.
            // This `sum` syntax creates a temporary result on device 1,
            // then `asElement` causes the temporary to be transferred to
            // the host, the value is retrieved, and the temp is released.
            // This syntax is good for experiments, but should not be used
            // for repetitive actions
            var sum = try using(device1) {
                try matrix.sum().asElement()
            }
            XCTAssert(sum == 15.0)

            // copy the matrix and simulate a readOnly operation on device2
            // a device array is allocated on device 2 then the master copy
            // on device 1 is copied to device 2.
            // Since device 1 and 2 are in the same service, a device to device
            // async copy is performed. In the case of Cuda, it would travel
            // across nvlink and not the PCI bus
            let matrix2 = matrix
            _ = try matrix2.readOnly(using: queue2)
            
            // copy matrix2 and simulate a readWrite operation on device2
            // this causes copy on write and mutate on device
            var matrix3 = matrix2
            _ = try matrix3.readWrite(using: queue2)

            // sum device 1 copy should be 15
            // `sum` creates a temp result tensor, allocates an array on
            // device 2, and performs the reduction.
            // Then `asElement` causes a host array to be allocated, and the
            // the data is copied from device 2 to host, the value is returned
            // and the temporary tensor is released.
            sum = try using(device2) {
                try matrix.sum().asElement()
            }
            XCTAssert(sum == 15.0)

            // matrix is overwritten with a new array on device 1
            matrix = using(device1) {
                matrix.filledWithIndex()
            }
            
            // sum matrix on device 2
            // `sum` creates a temporary result tensor on device 2
            // a device array for `matrix` is allocated on device 2 and
            // the matrix data is copied from device 1 to device 2
            // then `asElement` creates a host array and the result is
            // copied from device 2 to the host array, and then the tensor
            // is released.
            sum = try using(device2) {
                try matrix.sum().asElement()
            }
            XCTAssert(sum == 15.0)

            // exiting the scopy, matrix and matrix2 are released along
            // with all resources on all devices.
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //--------------------------------------------------------------------------
    // test_copyOnWriteDevice
    func test_copyOnWriteDevice() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
            
            // create a named queue on two different discreet devices
            // cpu devices 1 and 2 are discreet memory versions for testing
            let device1 = Platform.testCpu1

            // fill with index on device 1
            let index = (1, 1)
            var matrix1 = Matrix<Float>((3, 2))
            using(device1) {
                fillWithIndex(&matrix1)
            }
            // testing a value causes the data to be copied to the host
            var value = try matrix1.value(at: index)
            XCTAssert(value == 3.0)
            
            // copy and mutate data
            // the data will be duplicated wherever the source is
            var matrix2 = matrix1
            value = try matrix2.value(at: index)
            XCTAssert(value == 3.0)
            
            // writing to matrix2 causes view mutation and copy on write
            try matrix2.set(value: 7, at: index)
            value = try matrix1.value(at: index)
            XCTAssert(value == 3.0)
            
            value = try matrix2.value(at: index)
            XCTAssert(value == 7.0)
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //--------------------------------------------------------------------------
    // test_copyOnWriteCrossDevice
    func test_copyOnWriteCrossDevice() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
            
            // create a named queue on two different discreet devices
            // cpu devices 1 and 2 are discreet memory versions for testing
            let device1 = Platform.testCpu1
            let queue1 = device1.queues[0]
            let device2 = Platform.testCpu2
            let queue2 = device2.queues[0]

            let index = (1, 1)
            var matrix1 = Matrix<Float>((3, 2))

            // allocate array on device 1 and fill with indexes
            using(device1) {
                fillWithIndex(&matrix1)
            }
            
            // getting a value causes the data to be copied to an
            // array associated with the app thread
            // The master version is stil on device 1
            let value = try matrix1.value(at: index)
            print(value)
            XCTAssert(value == 3.0)

            // simulate read only access on device 1 and 2
            // data will be copied to device 2 for the first time
            _ = try matrix1.readOnly(using: queue1)
            _ = try matrix1.readOnly(using: queue2)

            // sum device 1 copy should be 15
            let sum1 = try using(device1) {
                try matrix1.sum().asElement()
            }
            XCTAssert(sum1 == 15.0)

            // clear the device 0 master copy
            using(device1) {
                fill(&matrix1, with: 0)
            }

            // sum device 1 copy should now also be 0
            // sum device 1 copy should be 15
            let sum2 = try using(device2) {
                try matrix1.sum().asElement()
            }
            XCTAssert(sum2 == 0)
            
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //--------------------------------------------------------------------------
    // test_copyOnWrite
    // NOTE: uses the default queue
    func test_copyOnWrite() {
        do {
            Platform.log.level = .diagnostic
//            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
            
            let index = (1, 1)
            var matrix1 = Matrix<Float>((3, 2))
            fillWithIndex(&matrix1)
            var value = try matrix1.value(at: index)
            XCTAssert(value == 3.0)
            
            var matrix2 = matrix1
            value = try matrix2.value(at: index)
            XCTAssert(value == 3.0)
            
            try matrix2.set(value: 7, at: index)
            value = try matrix1.value(at: index)
            XCTAssert(value == 3.0)
            
            value = try matrix2.value(at: index)
            XCTAssert(value == 7.0)
        } catch {
            XCTFail(String(describing: error))
        }
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
