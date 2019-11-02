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
//==============================================================================
// map
public extension Sequence {
    /// map a sequence to a tensor
    @inlinable
    func map<R>(to result: inout R,
                _ transform: (Element) -> R.MutableValues.Element) throws where
        R: TensorView
    {
        var iterator = self.makeIterator()
        var results = try result.mutableValues()
        
        for i in results.indices {
            if let value = iterator.next() {
                results[i] = transform(value)
            }
        }
    }
    
    /// map to a mutable collection
    @inlinable
    func map<R>(to result: inout R,
                _ transform: (Element) -> R.Element) where
        R: MutableCollection
    {
        
        var iterator = self.makeIterator()
        for i in result.indices {
            if let value = iterator.next() {
                result[i] = transform(value)
            }
        }
    }
}

//==============================================================================
public extension Zip2Sequence {
    typealias Pair = (Sequence1.Element, Sequence2.Element)
    
    /// map tensors
    @inlinable
    func map<T>(to result: inout T,
                _ transform: (Pair) -> T.MutableValues.Element) throws
        where T: TensorView
    {
        var iterator = self.makeIterator()
        var results = try result.mutableValues()
        
        for i in results.indices {
            if let pair = iterator.next() {
                results[i] = transform(pair)
            }
        }
        
    }
    
    /// map to a mutable collection
    @inlinable
    func map<Result>(to result: inout Result,
                     _ transform: (Pair) -> Result.Element)
        where Result: MutableCollection
    {
        var iterator = self.makeIterator()
        for i in result.indices {
            if let pair = iterator.next() {
                result[i] = transform(pair)
            }
        }
    }
}

//==============================================================================
public extension Zip3Sequence {
    typealias Input = (S1.Element, S2.Element, S3.Element)
    
    /// map tensors
    @inlinable
    func map<T>(to result: inout T,
                _ transform: (Input) -> T.MutableValues.Element) throws
        where T: TensorView
    {
        var iterator = self.makeIterator()
        var results = try result.mutableValues()
        
        for i in results.indices {
            if let input = iterator.next() {
                results[i] = transform(input)
            }
        }
        
    }
    
    /// map to a mutable collection
    @inlinable
    func map<Result>(to result: inout Result,
                     _ transform: (Input) -> Result.Element)
        where Result: MutableCollection
    {
        var iterator = self.makeIterator()
        for i in result.indices {
            if let input = iterator.next() {
                result[i] = transform(input)
            }
        }
    }
}

//==============================================================================
// zip
public func zip<T1, T2>(_ t1: T1, _ t2: T2) throws ->
    Zip2Sequence<TensorValueCollection<T1>, TensorValueCollection<T2>>
    where T1: TensorView, T2: TensorView
{
    return try zip(t1.values(), t2.values())
}

//==============================================================================
// reduce
public extension Sequence {
    /// reduce to a tensor
    func reduce<T>(
        to result: inout T,
        _ initialResult: Element,
        _ nextPartialResult: (Element, Element) throws -> Element) throws
        where T: TensorView, Element == T.Element
    {
        var results = try result.mutableValues()
        var partial = initialResult
        for value in self {
            partial = try nextPartialResult(partial, value)
        }
        results[results.startIndex] = partial
    }
    
    /// reduce to a mutable collection
    @inlinable
    func reduce<T>(
        to result: inout T,
        _ initialResult: Element,
        _ nextPartialResult: (Element, Element) throws -> Element) rethrows
        where T: MutableCollection, Element == T.Element
    {
        var partial = initialResult
        for value in self {
            partial = try nextPartialResult(partial, value)
        }
        result[result.startIndex] = partial
    }
}
