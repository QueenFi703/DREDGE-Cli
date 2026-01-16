// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "DREDGE-Cli",
    platforms: [
        .macOS(.v12),
        .iOS(.v15)
    ],
    products: [
        .library(
            name: "DredgeCore",
            targets: ["DredgeCore"]
        ),
        .executable(
            name: "dredge-cli",
            targets: ["DredgeCLI"]
        )
    ],
    targets: [
        .target(
            name: "DredgeCore",
            path: "Sources/DredgeCore"
        ),
        .target(
            name: "DredgeApp",
            dependencies: ["DredgeCore"],
            path: "Sources/DredgeApp",
            resources: [
                .process("AboutStrings.strings")
            ]
        ),
        .executableTarget(
            name: "DredgeCLI",
            dependencies: ["DredgeCore"],
            path: "Sources/DredgeCLI"
        ),
        .testTarget(
            name: "DredgeCoreTests",
            dependencies: ["DredgeCore"],
            path: "Tests/DredgeCoreTests"
        )
    ]
)
