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

#if canImport(CVulkan)
import CVulkan
#endif

@testable import SwiftRT

class test_Syntax: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_simple", test_simple),
        ("test_setServiceProperties", test_setServiceProperties),
        ("test_appThreadZipMapReduce", test_appThreadZipMapReduce),
        ("test_repeatVector", test_repeatVector),
        ("test_repeatValue", test_repeatValue),
        ("test_sumView", test_sumView),
        ("test_transpose", test_transpose),
        ("test_queues", test_queues),
        ("test_structuredScalar", test_structuredScalar),
        ("test_withResultPlacement", test_withResultPlacement),
        ("test_logging", test_logging),
        ("test_zeroCopyFixedSizeVectorCast", test_zeroCopyFixedSizeVectorCast),
    ]
    
    //==========================================================================
    // test_simple
    // initialize a matrix with a sequence and take the sum
    func test_simple() {
        Platform.log.level = .diagnostic
        Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
        
        do {
            let matrix = Matrix<Float>((3, 5), with: 0..<15)
            print(matrix.formatted((2,0)))
            let sum = try matrix.sum().asElement()
            XCTAssert(sum == 105.0)
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_setServiceProperties
    // sets properties in the Platform `serviceProperties` dictionary
    // to be used during compute service initialization
    func test_setServiceProperties() {
        #if canImport(CVulkan)
        Platform.log.level = .diagnostic
        Platform.log.categories = [.properties]
        Platform.local.servicePriority = [vulkanServiceName]
        Platform.vulkan?.configuration = [
            .applicationName : "MyApp",
            .applicationVersion : 41,
            .engineName : "MyEngine",
            .engineVersion : 42,
            .apiVersion : VK_API_VERSION_1_0
        ]
        
        do {
            let matrix = Matrix<Float>((3, 5), with: 0..<15)
            let sum = try matrix.sum().asElement()
            XCTAssert(sum == 105.0)
        } catch {
            XCTFail(String(describing: error))
        }
        #endif
    }
    
    //==========================================================================
    // test_appThreadZipMapReduce
    func test_appThreadZipMapReduce() {
        do {
            // create two tensors and fill with indexes
            let a = Matrix<Float>((2, 3), with: 0..<6)
            let b = Matrix<Float>((2, 3), with: 6..<12)
            
            let absum = try zip(a, b).map { $0 + $1 }
            
            let expected: [Float] = [6, 8, 10, 12, 14, 16]
            XCTAssert(absum == expected)
            
            let dot = try zip(a, b).map(*).reduce(0, +)
            XCTAssert(dot == 145.0)
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatVector
    // No matter the extents, `matrix` only uses the shared storage
    // from `rowVector` and repeats it through indexing
    func test_repeatVector() {
        do {
            let rowVector = Matrix<Int32>((1, 5), with: 0..<5)
            let rmatrix = Matrix((5, 5), repeating: rowVector)
            
            print(rmatrix.formatted((2,0)))
            let rmatrixExp: [Int32] = [
                0, 1, 2, 3, 4,
                0, 1, 2, 3, 4,
                0, 1, 2, 3, 4,
                0, 1, 2, 3, 4,
                0, 1, 2, 3, 4,
            ]
            var values = try rmatrix.array()
            XCTAssert(values == rmatrixExp)
            
            let colVector = Matrix<Int32>((5, 1), with: 0..<5)
            let cmatrix = Matrix((5, 5), repeating: colVector)
            
            print(cmatrix.formatted((2,0)))
            let cmatrixExp: [Int32] = [
                0, 0, 0, 0, 0,
                1, 1, 1, 1, 1,
                2, 2, 2, 2, 2,
                3, 3, 3, 3, 3,
                4, 4, 4, 4, 4,
            ]
            values = try cmatrix.array()
            XCTAssert(values == cmatrixExp)
            
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatValue
    // No matter the extents, `volume` only uses the shared storage
    // from `value` and repeats it through indexing
    func test_repeatValue() {
        do {
            let value: Int32 = 42
            let volume = Volume<Int32>((2, 3, 10), repeating: Volume(value))
            print(volume.formatted((2,0)))
            
            let expected = [Int32](repeating: value,
                                   count: volume.shape.elementCount)
            let values = try volume.array()
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_sumView
    // Select and sum a 3D sub region
    // - initialize a volume using explicit extents
    // - fill with indexes on the default device
    // - create a sub view and take the sum on the device
    // - return the scalar value back to the app thread
    func test_sumView() {
        do {
            let volume = Volume<Int32>((3, 4, 5)).filledWithIndex()
            print(volume.formatted((2,0)))
            
            let view = volume.view(at: [1, 1, 1], extents: [2, 2, 2])
            print(view.formatted((2,0)))
            
            let viewSum = try sum(view).asElement()
            XCTAssert(viewSum == 312)
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    /// zero copy transpose
    func test_transpose() {
        do {
            let matrix = Matrix<Float>((2, 3), with: 0..<6)
            print(matrix.formatted((2,0)))
            
            let tmatrix = matrix.t
            print(tmatrix.formatted((2,0)))
            
            let expected: [Float] = [0, 3, 1, 4, 2, 5]
            
            let values = try tmatrix.array()
            XCTAssert(values == expected)
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_queues
    // create a named queue on two different discreet devices
    // <cpu devices 1 and 2 are discreet memory versions for testing>
    //
    // This also shows how the object tracker can be checked to see if
    // there are any retain cycles.
    func test_queues() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
            
            let device1 = Platform.testCpu1
            let device2 = Platform.testCpu2

            let volume = using(device1) {
                Volume<Int32>((3, 4, 5)).filledWithIndex()
            }
            let view = volume.view(at: [1, 1, 1], extents: [2, 2, 2])

            let viewSum = try using(device2) {
                try sum(view).asElement()
            }
            XCTAssert(viewSum == 312)
            
        } catch {
            XCTFail(String(describing: error))
        }
        
        if ObjectTracker.global.hasUnreleasedObjects {
            print(ObjectTracker.global.getActiveObjectReport())
            XCTFail("Retain cycle detected")
        }
    }
    
    //==========================================================================
    // test_structuredScalar
    // create a named queue on two different discreet devices
    // <cpu devices 1 and 2 are discreet memory versions for testing>
    func test_structuredScalar() {
        let sample = RGBA<UInt8>(0, 1, 2, 3)
        let matrix = Matrix<RGBA<UInt8>>((2, 3), repeating: Matrix(sample))
        let nhwc = NHWCTensor<UInt8>(matrix)
        print(nhwc.formatted((2, 0)))
    }
    
    //==========================================================================
    // test_withResultPlacement
    func test_withResultPlacement() {
        do {
            let volume = Volume<Int32>((3, 4, 5)).filledWithIndex()
            let view = volume.view(at: [1, 1, 1], extents: [2, 2, 2])
            
            var viewSum = Volume<Int32>((1, 1, 1))
            sum(view, result: &viewSum)
            
            let value = try viewSum.asElement()
            XCTAssert(value == 312)
            
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_logging
    // create a named queue on two different discreet devices
    // <cpu devices 1 and 2 are discreet memory versions for testing>
    func test_logging() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]

            let device1 = Platform.testCpu1
            let device2 = Platform.testCpu2

            let volume = using(device1) {
                Volume<Int32>((3, 4, 5)).filledWithIndex()
            }
            let view = volume.view(at: [1, 1, 1], extents: [2, 2, 2])

            let viewSum = try using(device2) {
                try sum(view).asElement()
            }
            XCTAssert(viewSum == 312)
            
        } catch {
            XCTFail(String(describing: error))
        }
    }

    //==========================================================================
    // test_zeroCopyFixedSizeVectorCast
    func test_zeroCopyFixedSizeVectorCast() {
        do {
            Platform.log.level = .diagnostic
            Platform.log.categories = [.dataAlloc, .dataCopy, .dataMutation]
            let pixels = [RGB<Float>(0, 0.5, 1),RGB<Float>(0.25, 0.5, 0.75)]
            let vector = Vector<RGB<Float>>(name: "pixels", elements: pixels)
            let nhwc = NHWCTensor<Float>(vector: vector)
            let values = try nhwc.array()
            let expected: [Float] = [0, 0.5, 1.0, 0.25, 0.5, 0.75]
            XCTAssert(values == expected)
            
        } catch {
            XCTFail(String(describing: error))
        }
    }
}
