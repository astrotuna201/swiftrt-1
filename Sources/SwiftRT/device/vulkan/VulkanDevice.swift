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

public class VulkanDevice : LocalComputeDevice {
    //--------------------------------------------------------------------------
    // conformance properties
    public private(set) var trackingId = 0
    public private (set) weak var service: ComputeService!
    public private(set) var queues = [DeviceQueue]()
    public var deviceArrayReplicaKey = Platform.nextDeviceArrayReplicaKey
    public let id: Int
    public var logInfo: LogInfo
    public let name: String
    public var timeout: TimeInterval?
    public let memoryAddressing: MemoryAddressing
    public var deviceErrorHandler: DeviceErrorHandler?
    public var limits: DeviceLimits
    public let memory: MemoryProperties
    public var _lastError: Error? = nil
    public var _errorMutex: Mutex = Mutex()

    // implementation specific properties
    public let physicalDevice: VkPhysicalDevice
    
    //--------------------------------------------------------------------------
    // initializers
    public init(service: VulkanService,
                physicalDevice: VkPhysicalDevice,
                deviceId: Int,
                logInfo: LogInfo,
                timeout: TimeInterval?) throws
    {
        self.service = service
        self.physicalDevice = physicalDevice
        self.id = deviceId
        self.logInfo = logInfo
        self.timeout = timeout
        self.memoryAddressing = .discreet // TODO: query this
        
        // query device limits
        var deviceProperties = VkPhysicalDeviceProperties()
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties)
        self.limits = DeviceLimits(limits: deviceProperties.limits)
        
        // get the device name
        let deviceNamePointer =
            withUnsafeBytes(of: &deviceProperties.deviceName) {
            return $0.baseAddress!.assumingMemoryBound(to: CChar.self)
        }
        self.name = String(cString: deviceNamePointer)
        
        //------------------------------------
        // TODO: get device memory properties
        let heaps = [MemoryHeap]()
        var memoryProperties = VkPhysicalDeviceMemoryProperties()
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties)
        
        let count = Int(memoryProperties.memoryTypeCount)
        _ = withUnsafeMutablePointer(to: &memoryProperties.memoryTypes) {
            $0.withMemoryRebound(to: VkMemoryType.self, capacity: count) {
                for i in 0..<count {
                    print(String(describing: MemoryAttributes(flags: $0[i].propertyFlags)))
                }
            }
        }
        self.memory = MemoryProperties(addressing: .discreet, heaps: heaps)

        // register device with the object tracker.
        // devices are statically held by the Platform.service
        trackingId = ObjectTracker.global
                .register(self, namePath: logNamePath, isStatic: true)
    }
    deinit { ObjectTracker.global.remove(trackingId: trackingId) }
    
    //--------------------------------------------------------------------------
    // createArray
    public func createArray(byteCount: Int, heapIndex: Int = 0, zero: Bool) throws
        -> DeviceArray
    {
//        return try CudaDeviceArray(device: self, count: count)
        fatalError()
    }

    //--------------------------------------------------------------------------
    // createReferenceArray
    /// creates a read only device array from a uma buffer
    public func createReferenceArray(buffer: UnsafeRawBufferPointer)
        -> DeviceArray
    {
        //        return CudaDeviceArray(device: self, buffer: buffer)
        fatalError()
    }

    //--------------------------------------------------------------------------
    // createMutableReferenceArray
    /// creates a read write device array from a uma buffer
    public func createMutableReferenceArray(
            buffer: UnsafeMutableRawBufferPointer) -> DeviceArray
    {
//        return CudaDeviceArray(device: self, buffer: buffer)
        fatalError()
    }

    //--------------------------------------------------------------------------
    // createStream
    public func createQueue(name: String, isStatic: Bool) throws -> DeviceQueue
    {
        fatalError()
    }
}

//==============================================================================
public extension DeviceLimits {
    init(limits: VkPhysicalDeviceLimits) {
        maxComputeSharedMemorySize = Int(limits.maxComputeSharedMemorySize)
        maxComputeWorkGroupCount =
            (Int(limits.maxComputeWorkGroupCount.0),
             Int(limits.maxComputeWorkGroupCount.1),
             Int(limits.maxComputeWorkGroupCount.2))
        maxComputeWorkGroupInvocations =
            Int(limits.maxComputeWorkGroupInvocations)
        maxComputeWorkGroupSize =
            (Int(limits.maxComputeWorkGroupSize.0),
             Int(limits.maxComputeWorkGroupSize.1),
             Int(limits.maxComputeWorkGroupSize.2))
        maxMemoryAllocationCount = Int(limits.maxMemoryAllocationCount)
    }
}

#endif
