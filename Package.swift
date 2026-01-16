// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "DREDGE-Cli",
    platforms: [
        .macOS(.v12),
        .iOS(.v15)
    ],
    products: [
        // CLI executable
        .executable(
            name: "dredge-cli",
            targets: ["DREDGECli"]
        ),
        // Shared core library
        .library(
            name: "DredgeCore",
            targets: ["DredgeCore"]
        )
    ],
    targets: [
        // CLI executable - ONLY includes swift/Sources/main.swift
        .executableTarget(
            name: "DREDGECli",
            path: "swift/Sources",
            exclude: []
        ),
        // Shared core library
        .target(
            name: "DredgeCore",
            path: "dredge-x-dolly/DredgeCore"
        ),
        // Tests
        .testTarget(
            name: "DREDGE-CliTests",
            dependencies: ["DREDGECli"],
            path: "swift/Tests/DREDGE-CliTests"
        )
    ]
)
