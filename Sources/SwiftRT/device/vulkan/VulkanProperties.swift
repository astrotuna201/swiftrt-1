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
// VK_MAKE_VERSION
// macros don't make it through the bridging process, so redefine it here
public func VK_MAKE_VERSION(_ major: Int, _ minor: Int, _ patch: Int) -> Int32 {
    return Int32((major << 22) | (minor << 12) | patch)
}

// Vulkan 1.0 version number
// Patch version should always be set to 0
public let VK_API_VERSION_1_0 = VK_MAKE_VERSION(1, 0, 0)
// Vulkan 1.1 version number
public let VK_API_VERSION_1_1 = VK_MAKE_VERSION(1, 1, 0)

#endif
