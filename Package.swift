// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "DREDGE-Cli",
    platforms: [
        .macOS(.v12),
        .iOS(.v15)
    ],
    products: [
        // Executable CLI tool
        .executable(
            name: "dredge-cli",
            targets: ["DREDGECli"]
        )
    ],
    targets: [
        // CLI executable target (from swift/Sources/DREDGECli.swift)
        .executableTarget(
            name: "DREDGECli",
            path: "swift/Sources"
        ),
        // Test target
        .testTarget(
            name: "DREDGE-CliTests",
            dependencies: ["DREDGECli"],
            path: "swift/Tests/DREDGE-CliTests"
        )
    ]
)
