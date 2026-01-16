// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "DREDGE-Cli",
    platforms: [
        .macOS(. v12)
    ],
    products: [
        .executable(
            name: "dredge-cli",
            targets: ["DREDGE-Cli"]
        )
    ],
    targets: [
        .executableTarget(
            name: "DREDGE-Cli",
            path: "Sources"
        ),
        .testTarget(
            name: "DREDGE-CliTests",
            dependencies: ["DREDGE-Cli"]
        )
    ]
)
