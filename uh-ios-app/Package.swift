// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "UHiOSApp",
    platforms: [
        .iOS(.v16),
        .macOS(.v13)
    ],
    products: [
        .library(
            name: "UHiOSApp",
            targets: ["UHiOSApp"]
        ),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "UHiOSApp",
            dependencies: []
        ),
    ]
)
