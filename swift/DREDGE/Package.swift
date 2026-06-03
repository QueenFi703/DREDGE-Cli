// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "dredge-swift",
    platforms: [
        .macOS(.v12),
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "DREDGE",
            targets: ["DREDGE"]
        )
    ],
    targets: [
        .target(
            name: "DREDGE",
            path: "Sources"
        ),
        .testTarget(
            name: "DREDGETests",
            dependencies: ["DREDGE"],
            path: "Tests"
        )
    ]
)
