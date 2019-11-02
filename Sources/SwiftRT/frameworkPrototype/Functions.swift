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
// >>>>>> INTENT <<<<<<
public extension DeviceFunctions {
    /// neg
    /// returns the element-wise negation
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Element: SignedNumeric
    {
        try! x.values().map(to: &result) { -$0 }
    }

    /// neg
    /// returns the element-wise negation
    func neg<T>(x: T) -> T where
        T: TensorView, T.Element: SignedNumeric
    {
//        return x.map { -$0 }
        fatalError()
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceFunctions])
// target generated from Intent by the compiler
public extension CpuQueue {
    /// neg
    func neg<T>(x: T, result: inout T) where
        T: TensorView, T.Element: SignedNumeric
    {
        queue(#function, { try x.values() }, &result) {
            $0.map(to: &$1) { -$0 }
        }
    }
}

//==============================================================================
// >>>>>> INTENT <<<<<<
public extension DeviceFunctions {
    /// fills the view with the scalar value
    func fill<T>(_ result: inout T, with value: T.Element) where T: TensorView {
        // TODO: can we hide the values/mutable values collections
        var values = try! result.mutableValues()
        for index in values.indices {
            values[index] = value
        }
    }
    
    /// fills the view with the spatial sequential index
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T: TensorView, T.Element: AnyNumeric
    {
        // TODO: can we hide the values/mutable values collections
        var value = startAt
        var values = try! result.mutableValues()
        for index in values.indices {
            values[index] = T.Element(any: value)
            value += 1
        }
    }
}

//******************************************************************************
// >>>>>> GENERATED <<<<<<
// @Target(type:"CPU", appliedTo:"CpuQueue", protocols:[DeviceFunctions])
// target generated from Intent by the compiler
public extension CpuQueue {
    //--------------------------------------------------------------------------
    /// fill(result:with:
    /// NOTE: this can be much faster, doesn't need to be ordered access
    func fill<T>(_ result: inout T, with value: T.Element) where T: TensorView {
        queue(#function, {}, &result) {
//            try result.readWrite().initialize(repeating: value)
            for index in $1.indices { $1[index] = value }
        }
    }
    
    //--------------------------------------------------------------------------
    /// fillWithIndex(x:startAt:
    func fillWithIndex<T>(_ result: inout T, startAt: Int) where
        T: TensorView, T.Element: AnyNumeric
    {
        queue(#function, {}, &result) {
            var value = startAt
            for index in $1.indices {
                $1[index] = T.Element(any: value)
                value += 1
            }
        }
    }
}

