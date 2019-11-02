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

class test_IterateView: XCTestCase {
    //==========================================================================
    // support terminal test run
    static var allTests = [
        ("test_Vector", test_Vector),
        ("test_Matrix", test_Matrix),
        ("test_Volume", test_Volume),
        ("test_VectorSubView", test_VectorSubView),
        ("test_MatrixSubView", test_MatrixSubView),
        ("test_VolumeSubView", test_VolumeSubView),
        ("test_perfVector", test_perfVector),
        ("test_perfMatrix", test_perfMatrix),
        ("test_perfVolume", test_perfVolume),
        ("test_perfIndexCopy", test_perfIndexCopy),
        ("test_repeatingValue", test_repeatingValue),
        ("test_repeatingRow", test_repeatingRow),
        ("test_repeatingCol", test_repeatingCol),
        ("test_repeatingColInVolume", test_repeatingColInVolume),
        ("test_repeatingMatrix", test_repeatingMatrix),
        ("test_repeatingMatrixSubView", test_repeatingMatrixSubView),
    ]
    
    //==========================================================================
    // test_Vector
    func test_Vector() {
        do {
            let count: Int32 = 10
            let expected = [Int32](0..<count)
            let vector = Vector<Int32>(elements: expected)
            print(vector.formatted((2,0)))
            
            let values = try vector.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_Matrix
    func test_Matrix() {
        do {
            let expected = [Int32](0..<4)
            let matrix = Matrix<Int32>((2, 2), elements: expected)
//                        print(matrix.formatted((2,0)))
            
            let values = try matrix.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_Volume
    func test_Volume() {
        do {
            let expected = [Int32](0..<24)
            let volume = Volume<Int32>((2, 3, 4), elements: expected)
            //            print(volume.formatted((2,0)))
            
            let values = try volume.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_VectorSubView
    func test_VectorSubView() {
        do {
            let vector = Vector<Int32>(with: 0..<10)
            let view = vector.view(at: [2], extents: [3])
            //            print(subView.formatted((2,0)))
            
            let expected: [Int32] = [2, 3, 4]
            let values = try view.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_MatrixSubView
    func test_MatrixSubView() {
        do {
            let matrix = Matrix<Int32>((3, 4), with: 0..<12)
            let view = matrix.view(at: [1, 1], extents: [2, 2])
                    print(view.formatted((2,0)))
            
            let expected: [Int32] = [
                5, 6,
                9, 10
            ]
            let values = try view.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_VolumeSubView
    func test_VolumeSubView() {
        do {
            let volume = Volume<Int32>((3, 3, 4), with: 0..<36)
            let view = volume.view(at: [1, 1, 1], extents: [2, 2, 3])
//            print(view.formatted((2,0)))
            
            let expected: [Int32] = [
                17, 18, 19,
                21, 22, 23,
                
                29, 30, 31,
                33, 34, 35,
            ]
            let values = try view.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_perfVector
    func test_perfVector() {
        #if !DEBUG
        do {
            let count = 1024 * 1024
            let vector = Vector<Int32>(any: 0..<count)
            //            print(matrix.formatted((2,0)))
            let values = try vector.values()
            
            self.measure {
                for _ in values {}
            }
        } catch {
            XCTFail(String(describing: error))
        }
        #endif
    }
    
    //==========================================================================
    // test_perfMatrix
    func test_perfMatrix() {
        #if !DEBUG
        do {
            let rows = 1024
            let cols = 1024

            let matrix = Matrix<Int32>((rows, cols), any: 0..<(rows * cols))
            //            print(matrix.formatted((2,0)))
            
            let values = try matrix.values()

            self.measure {
                for _ in values {}
            }
        } catch {
            XCTFail(String(describing: error))
        }
        #endif
    }
    
    //==========================================================================
    // test_perfVolume
    func test_perfVolume() {
        #if !DEBUG
        do {
            let depths = 4
            let rows = 512
            let cols = 512
            
            let matrix = Volume<Int32>((depths, rows, cols),
                                       any: 0..<(depths * rows * cols))
            //            print(matrix.formatted((2,0)))
            
            let values = try matrix.values()
            
            self.measure {
                for _ in values {}
            }
        } catch {
            XCTFail(String(describing: error))
        }
        #endif
    }
    
    //==========================================================================
    // test_perfIndexCopy
    func test_perfIndexCopy() {
        #if !DEBUG
        var m = Matrix<Int32>((1024, 1024)).startIndex
        
        self.measure {
            for _ in 0..<1000000 {
                m = m.increment()
//                m = m.advanced(by: 1)
            }
        }
        #endif
    }
    //==========================================================================
    // test_repeatingValue
    func test_repeatingValue() {
        do {
            // try repeating a scalar
            let value = Matrix<Int32>((1, 1), elements: [42])
            let matrix = Matrix<Int32>((2, 3), repeating: value)
            //            print(vector.formatted((2,0)))
            
            let expected: [Int32] = [
                42, 42, 42,
                42, 42, 42,
            ]
            
            let values = try matrix.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatingRow
    func test_repeatingRow() {
        do {
            // try repeating a row vector
            let row = Matrix<Int32>((1, 3), with: 0..<3)
            let matrix = Matrix<Int32>((2, 3), repeating: row)
            //            print(matrix.formatted((2,0)))
            
            let expected: [Int32] = [
                0, 1, 2,
                0, 1, 2,
            ]
            
            let values = try matrix.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatingCol
    func test_repeatingCol() {
        do {
            // try repeating a row vector
            let col = Matrix<Int32>((3, 1), with: 0..<3)
            let matrix = Matrix<Int32>((3, 2), repeating: col)
                        print(matrix.formatted((2,0)))
            
            let expected: [Int32] = [
                0, 0,
                1, 1,
                2, 2,
            ]
            
            let values = try matrix.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatingColInVolume
    func test_repeatingColInVolume() {
        do {
            let pattern = Volume<Int32>((1, 3, 1), elements: [
                1,
                0,
                1,
            ])
            
            let matrix = Volume<Int32>((2, 3, 4), repeating: pattern)
            let expected: [Int32] = [
                1, 1, 1, 1,
                0, 0, 0, 0,
                1, 1, 1, 1,
                
                1, 1, 1, 1,
                0, 0, 0, 0,
                1, 1, 1, 1,
            ]
            
            let values = try matrix.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatingMatrix
    func test_repeatingMatrix() {
        do {
            let pattern = Volume<Int32>((1, 3, 4), elements: [
                1, 0, 1, 0,
                0, 1, 0, 1,
                1, 0, 1, 0,
            ])
            
            let matrix = Volume<Int32>((2, 3, 4), repeating: pattern)
            let expected: [Int32] = [
                1, 0, 1, 0,
                0, 1, 0, 1,
                1, 0, 1, 0,

                1, 0, 1, 0,
                0, 1, 0, 1,
                1, 0, 1, 0,
            ]

            let values = try matrix.array()
            XCTAssert(values == expected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
    //==========================================================================
    // test_repeatingMatrixSubView
    func test_repeatingMatrixSubView() {
        do {
            let pattern = Matrix<Int32>((3, 1), elements: [
                1,
                0,
                1,
            ])
            
            let matrix = Matrix<Int32>((3, 4), repeating: pattern)
            let expected: [Int32] = [
                1, 1, 1, 1,
                0, 0, 0, 0,
                1, 1, 1, 1,
            ]

            let values = try matrix.array()
            XCTAssert(values == expected, "values do not match")

            let view = matrix.view(at: [1, 1], extents: [2, 3])
            let viewExpected: [Int32] = [
                0, 0, 0,
                1, 1, 1,
            ]

            let viewValues = try view.array()
            XCTAssert(viewValues == viewExpected, "values do not match")
        } catch {
            XCTFail(String(describing: error))
        }
    }
    
}
