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
// VulkanService
public final class VulkanService: LocalComputeService {
    // service protocol configuration
    public private(set) weak var platform: ComputePlatform!
    public private(set) var trackingId = 0
    public private(set) var devices = [ComputeDevice]()
    public var deviceErrorHandler: DeviceErrorHandler?
    public var _lastError: Error?
    public var _errorMutex: Mutex = Mutex()
    public let id: Int
    public var logInfo: LogInfo
    public let name: String
    public var configuration: [VulkanPropertyKey: Any]

    // vulkan specific configuration
    private var instance: VkInstance!
    private var debugReportCallback: VkDebugReportCallbackEXT!
    
    //--------------------------------------------------------------------------
    // timeout
    public var timeout: TimeInterval? {
        didSet {
            devices.forEach {
                $0.timeout = timeout
            }
        }
    }

    //--------------------------------------------------------------------------
    // initializers
    public required init(platform: ComputePlatform,
                         id: Int,
                         logInfo: LogInfo,
                         name: String? = nil) throws {
        self.platform = platform
        self.id = id
        self.name = name ?? "vulkan"
        self.logInfo = logInfo
        configuration = [VulkanPropertyKey: Any]()
        
        // create the vulkan instance
        instance = try createVkInstance()
        
        // create a ComputeDevice for each physical vulkan device
        try createComputeDevices()
        
        // this is held statically by the Platform
        trackingId = ObjectTracker.global.register(self, isStatic: true)
    }

    deinit {
        ObjectTracker.global.remove(trackingId: trackingId)
    }

    //--------------------------------------------------------------------------
    // createVkInstance
    // Creates the VkInstance object, which is the root of the environment.
    // During initialization the `Platform.serviceProperties` are checked for
    // user specified configuration property values.
    private func createVkInstance() throws -> VkInstance {
        // list of enabled layers for validation and extensions
        var enabledLayers = [CStringPointer?]()
        var enabledExtensions = [CStringPointer?]()
        
        //-----------------------------------
        // Enable Vulkan validation layers in DEBUG mode
        #if DEBUG
        // query supported validation layers to find "standard_validation"
        var layerCount: UInt32 = 0
        try vkCheck(vkEnumerateInstanceLayerProperties(&layerCount, nil))

        var layerProps = [VkLayerProperties](repeating: VkLayerProperties(),
                                             count: Int(layerCount))
        try vkCheck(vkEnumerateInstanceLayerProperties(&layerCount,
                                                       &layerProps))
        
        let stdName = "VK_LAYER_LUNARG_standard_validation"
        for i in 0..<layerProps.count {
            // point to retained string tuple
            let layerNamePointer = withUnsafeBytes(of: &layerProps[i].layerName)
            {
                return $0.baseAddress!.assumingMemoryBound(to: CChar.self)
            }
            if String(cString: layerNamePointer) == stdName {
                // retain the C pointer to be handed back later
                // this is safe because layerProps is in the same scope
                enabledLayers.append(layerNamePointer)
                break
            }
        }
        if enabledLayers.isEmpty {
            writeLog("Layer \(stdName) not supported", level: .warning)
        }
        
        // Enable VK_EXT_DEBUG_REPORT_EXTENSION_NAME extension so that
        // validation layers will emit warnings
        var extensionCount: UInt32 = 0
        try vkCheck(vkEnumerateInstanceExtensionProperties(nil,
                                                           &extensionCount,
                                                           nil))
        var extensionProps =
            [VkExtensionProperties](repeating: VkExtensionProperties(),
                                    count: Int(extensionCount))
        try vkCheck(vkEnumerateInstanceExtensionProperties(nil,
                                                           &extensionCount,
                                                           &extensionProps))
        let extName = VK_EXT_DEBUG_REPORT_EXTENSION_NAME
        for i in 0..<extensionProps.count {
            // point to retained string tuple
            let extensionNamePointer =
                withUnsafeBytes(of: &extensionProps[i].extensionName) {
                return $0.baseAddress!.assumingMemoryBound(to: CChar.self)
            }
            
            if String(cString: extensionNamePointer) == extName {
                // retain the C pointer to be handed back later
                // this is safe because extensionProps is in the same scope
                enabledExtensions.append(extensionNamePointer)
                break
            }
        }
        if enabledExtensions.isEmpty {
            writeLog("Extension \(extName) not supported", level: .warning)
        }
        #endif

        //-----------------------------------
        // get optional application name from vulkan configuration
        var cApplicationName = mallocPropertyCString(for: .applicationName)
        defer { free(cApplicationName) }
        let applicationVersion =
            UInt32((configuration[.applicationVersion] as? Int) ?? 0)
        
        // get optional engine name from vulkan configuration
        var cEngineName = mallocPropertyCString(for: .engineName)
        defer { free(cEngineName) }
        
        let engineVersion =
            UInt32((configuration[.engineVersion] as? Int) ?? 0)
        
        let apiVersion = UInt32((configuration[.apiVersion] as? Int32) ??
            VK_API_VERSION_1_1)

        // initialize the VkApplicationInfo.
        var applicationInfo = VkApplicationInfo(
            sType: VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pNext: nil,
            pApplicationName: cApplicationName,
            applicationVersion: applicationVersion,
            pEngineName: cEngineName,
            engineVersion: engineVersion,
            apiVersion: apiVersion)
        
        var createInfo = VkInstanceCreateInfo(
            sType: VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pNext: nil,
            flags: 0,
            pApplicationInfo: &applicationInfo,
            enabledLayerCount: UInt32(enabledLayers.count),
            ppEnabledLayerNames: &enabledLayers,
            enabledExtensionCount: UInt32(enabledExtensions.count),
            ppEnabledExtensionNames: &enabledExtensions)
        
        // vkCreateInstance will throw if it failed, so result can't be nil
        var instance: VkInstance!
        try vkCheck(vkCreateInstance(&createInfo, nil, &instance))
        
        //-----------------------------------
        // Register a callback function for the extension
        // VK_EXT_DEBUG_REPORT_EXTENSION_NAME, so that warnings emitted
        // from the validation layer are written to the log
        #if DEBUG
        func debugCallback(
            flags: VkDebugReportFlagsEXT,
            objectType: VkDebugReportObjectTypeEXT,
            object: UInt64,
            location: Int,
            messageCode: Int32,
            pLayerPrefix: UnsafePointer<Int8>?,
            pMessage: UnsafePointer<Int8>?,
            pUserData: UnsafeMutableRawPointer?) -> VkBool32
        {
            Platform.local.writeLog("\(String(cString: pLayerPrefix!)): " +
                "\(String(cString: pMessage!))")
            return VkBool32(VK_FALSE)
        }

        // load the callback report function pointer
        let cCallbackFnName = strdup("vkCreateDebugReportCallbackEXT")
        defer { free(cCallbackFnName) }
        
        let vkCreateDebugReportCallbackEXT =
            unsafeBitCast(vkGetInstanceProcAddr(instance!, cCallbackFnName),
                          to: PFN_vkCreateDebugReportCallbackEXT.self)

        // init the report create info
        let flags: VkDebugReportFlagsEXT =
            VK_DEBUG_REPORT_ERROR_BIT_EXT.rawValue |
                VK_DEBUG_REPORT_WARNING_BIT_EXT.rawValue |
                VK_DEBUG_REPORT_PERFORMANCE_WARNING_BIT_EXT.rawValue
        
        var reportCreateInfo = VkDebugReportCallbackCreateInfoEXT(
            sType: VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT,
            pNext: nil,
            flags: flags,
            pfnCallback: debugCallback,
            pUserData: nil)
        
        // Register callback
        try vkCheck(vkCreateDebugReportCallbackEXT(instance,
                                                   &reportCreateInfo,
                                                   nil, // allocator
                                                   &debugReportCallback))
        #endif

        // done
        return instance
    }
    
    //--------------------------------------------------------------------------
    // mallocPropertyCString
    // retrieves a String property value from the Platform.configuration
    // dictionary and converts the value into a CString.
    // NOTE: It is the callers responsibility to `free` the CString after use
    private func mallocPropertyCString(for property: VulkanPropertyKey) ->
        UnsafeMutablePointer<Int8>?
    {
        if let value = configuration[property] {
            assert(value is String, "\(property) must be of type String")
            return strdup(value as! String)
        } else {
            return nil
        }
    }

    //--------------------------------------------------------------------------
    // createComputeDevices
    // Create a ComputeDevice for each physical vulkan device
    private func createComputeDevices() throws {
        // get the device count
        var deviceCount: UInt32 = 0
        try vkCheck(vkEnumeratePhysicalDevices(instance, &deviceCount, nil))
        if deviceCount == 0 {
            writeLog("No vulkan devices found", level: .warning)
        }
        
        // get the physical device list
        var physicalDevices = [VkPhysicalDevice?](repeating: nil,
                                                  count: Int(deviceCount))
        try vkCheck(vkEnumeratePhysicalDevices(instance, &deviceCount,
                                               &physicalDevices))
        
        //-----------------------------------
        // create a logical device for each physical device
        for (i, physicalDevice) in physicalDevices.enumerated() {
            // create logical device
//            let logicalDevice =
              _ =  try VulkanDevice(service: self,
                                 physicalDevice: physicalDevice!,
                                 deviceId: i,
                                 logInfo: logInfo.flat("dev:\(i)"),
                                 timeout: timeout)
        }
    }
}

//==============================================================================
/// a set of predefined property names to simplify configuring
/// the service properties
public enum VulkanPropertyKey: Int {
    case applicationName
    case applicationVersion
    case engineName
    case engineVersion
    case apiVersion
}

#endif
