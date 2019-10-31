// swift-tools-version:5.1
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

#if os(Linux)
import Glibc
#else
import Darwin.C
#endif

var products: [PackageDescription.Product] = [
    .library(name: "SwiftRT", targets: ["SwiftRT"])]
var dependencies: [Target.Dependency] = []
var targets: [PackageDescription.Target] = []

if getenv("SWIFTRT_ENABLE_VULKAN") != nil {
    products.append(.library(name: "CVulkan", targets: ["CVulkan"]))
    dependencies.append("CVulkan")
    targets.append(
        .systemLibrary(name: "CVulkan",
                       path: "Libraries/Vulkan",
                       pkgConfig: "vulkan"))
}

if getenv("SWIFTRT_ENABLE_CUDA") != nil {
    products.append(.library(name: "CCuda", targets: ["CCuda"]))
    dependencies.append("CCuda")
    targets.append(
        .systemLibrary(name: "CCuda",
                       path: "Libraries/Cuda",
                       pkgConfig: "cuda"))
}

targets.append(contentsOf: [
    .target(name: "SwiftRT",
        dependencies: dependencies),
    .testTarget(name: "SwiftRTTests",
            dependencies: ["SwiftRT"]),
])

let package = Package(
    name: "SwiftRT",
    products: products,
    dependencies: [],
    targets: targets
)
