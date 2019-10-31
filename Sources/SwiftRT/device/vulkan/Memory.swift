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

#if canImport(CVulkan)
import CVulkan

//==============================================================================
/// MemoryPropertyFlagOptions
/// an OptionSet that represents VkMemoryPropertyFlags
//public struct MemoryAttributes: OptionSet, CustomStringConvertible {
//    public let rawValue: UInt32
//
//    public init(rawValue: UInt32) { self.rawValue = rawValue }
//
//    static let deviceLocal = MemoryAttributes(rawValue: VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.rawValue)
//    static let hostVisible = MemoryAttributes(rawValue: VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT.rawValue)
//    static let hostCoherent = MemoryAttributes(rawValue: VK_MEMORY_PROPERTY_HOST_COHERENT_BIT.rawValue)
//    static let hostCached = MemoryAttributes(rawValue: VK_MEMORY_PROPERTY_HOST_CACHED_BIT.rawValue)
//    static let lazilyAllocated = MemoryAttributes(rawValue: VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT.rawValue)
//    static let protected = MemoryAttributes(rawValue: VK_MEMORY_PROPERTY_PROTECTED_BIT.rawValue)
//    static let deviceCoherentAMD = MemoryAttributes(rawValue: VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD.rawValue)
//    static let deviceUncachedAMD = MemoryAttributes(rawValue: VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD.rawValue)
//
//    public var description: String {
//        var string = "["
//        if self.contains(.deviceLocal) { string += ".deviceLocal, " }
//        if self.contains(.hostVisible) { string += ".hostVisible, " }
//        if self.contains(.hostCoherent) { string += ".hostCoherent, " }
//        if self.contains(.hostCached) { string += ".hostCached, " }
//        if self.contains(.lazilyAllocated) { string += ".lazilyAllocated, " }
//        if self.contains(.protected) { string += ".protected, " }
//        if self.contains(.deviceCoherentAMD) { string += ".deviceCoherentAMD, " }
//        if self.contains(.deviceUncachedAMD) { string += ".deviceUncachedAMD, " }
//        string.removeLast(2)
//        string += "]"
//        return string
//    }
//}

extension MemoryAttributes {
    init(flags: VkMemoryPropertyFlags) {
        var value = 0
        if flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT.rawValue != 0 {
            value |= MemoryAttributes.deviceLocal.rawValue
        }
        self = MemoryAttributes(rawValue: value)
    }
}

//==============================================================================
/// LocalComputeDevice queryMemoryBudget
extension LocalComputeDevice {
    public func queryMemoryBudget() -> MemoryBudget {
        fatalError()
    }
}

#endif
