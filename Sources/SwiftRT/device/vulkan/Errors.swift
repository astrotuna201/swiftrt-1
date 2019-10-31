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

#if canImport(CVulkan)
import CVulkan

//==============================================================================
// vkCheck
public func vkCheck(_ status: VkResult, file: String = #file,
                    function: String = #function, line: Int = #line) throws {
    if status != VK_SUCCESS {
        let location = "Vulkan error in \(file) at \(function):\(line)"
        throw ServiceError.functionFailure(location: location,
                                           message: String(describing: status))
    }
}

//==============================================================================
// VkResult extensions
extension VkResult: Hashable, CustomStringConvertible {
    public var description: String {
        return errorStrings[self] ?? "Unknown Vulkan Error code(\(self))"
    }
}

private let errorStrings = [
    VK_SUCCESS: "VK_SUCCESS",
    VK_NOT_READY: "VK_NOT_READY",
    VK_TIMEOUT: "VK_TIMEOUT",
    VK_EVENT_SET: "VK_EVENT_SET",
    VK_EVENT_RESET: "VK_EVENT_RESET",
    VK_INCOMPLETE: "VK_INCOMPLETE",
    VK_ERROR_OUT_OF_HOST_MEMORY: "VK_ERROR_OUT_OF_HOST_MEMORY",
    VK_ERROR_OUT_OF_DEVICE_MEMORY: "VK_ERROR_OUT_OF_DEVICE_MEMORY",
    VK_ERROR_INITIALIZATION_FAILED: "VK_ERROR_INITIALIZATION_FAILED",
    VK_ERROR_DEVICE_LOST: "VK_ERROR_DEVICE_LOST",
    VK_ERROR_MEMORY_MAP_FAILED: "VK_ERROR_MEMORY_MAP_FAILED",
    VK_ERROR_LAYER_NOT_PRESENT: "VK_ERROR_LAYER_NOT_PRESENT",
    VK_ERROR_EXTENSION_NOT_PRESENT: "VK_ERROR_EXTENSION_NOT_PRESENT",
    VK_ERROR_FEATURE_NOT_PRESENT: "VK_ERROR_FEATURE_NOT_PRESENT",
    VK_ERROR_INCOMPATIBLE_DRIVER: "VK_ERROR_INCOMPATIBLE_DRIVER",
    VK_ERROR_TOO_MANY_OBJECTS: "VK_ERROR_TOO_MANY_OBJECTS",
    VK_ERROR_FORMAT_NOT_SUPPORTED: "VK_ERROR_FORMAT_NOT_SUPPORTED",
    VK_ERROR_FRAGMENTED_POOL: "VK_ERROR_FRAGMENTED_POOL",
    VK_ERROR_OUT_OF_POOL_MEMORY: "VK_ERROR_OUT_OF_POOL_MEMORY",
    VK_ERROR_INVALID_EXTERNAL_HANDLE: "VK_ERROR_INVALID_EXTERNAL_HANDLE",
    VK_ERROR_SURFACE_LOST_KHR: "VK_ERROR_SURFACE_LOST_KHR",
    VK_ERROR_NATIVE_WINDOW_IN_USE_KHR: "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR",
    VK_SUBOPTIMAL_KHR: "VK_SUBOPTIMAL_KHR",
    VK_ERROR_OUT_OF_DATE_KHR: "VK_ERROR_OUT_OF_DATE_KHR",
    VK_ERROR_INCOMPATIBLE_DISPLAY_KHR: "VK_ERROR_INCOMPATIBLE_DISPLAY_KHR",
    VK_ERROR_VALIDATION_FAILED_EXT: "VK_ERROR_VALIDATION_FAILED_EXT",
    VK_ERROR_INVALID_SHADER_NV: "VK_ERROR_INVALID_SHADER_NV",
    VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT: "VK_ERROR_INVALID_DRM_FORMAT_MODIFIER_PLANE_LAYOUT_EXT",
    VK_ERROR_FRAGMENTATION_EXT: "VK_ERROR_FRAGMENTATION_EXT",
    VK_ERROR_NOT_PERMITTED_EXT: "VK_ERROR_NOT_PERMITTED_EXT",
    VK_ERROR_INVALID_DEVICE_ADDRESS_EXT: "VK_ERROR_INVALID_DEVICE_ADDRESS_EXT",
    VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT: "VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT",
    
    // These are overloaded in the vulkan header for some reason,
    // so these VkResult values are unfortunately ambiguous.
    // VK_ERROR_OUT_OF_POOL_MEMORY_KHR = VK_ERROR_OUT_OF_POOL_MEMORY,
    // VK_ERROR_OUT_OF_POOL_MEMORY_KHR: "VK_ERROR_OUT_OF_POOL_MEMORY_KHR",
    // VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR = VK_ERROR_INVALID_EXTERNAL_HANDLE,
    // VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR: "VK_ERROR_INVALID_EXTERNAL_HANDLE_KHR",
]

#endif
