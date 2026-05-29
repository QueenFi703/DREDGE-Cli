// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "DREDGE-Cli",
    platforms: [
        .macOS(.v12),
        .iOS(.v15)
    ],
    products: [
        .executable(
            name: "dredge-cli",
            targets: ["DREDGECli"]
        ),
        .library(
            name: "DREDGEMVPApp",
            targets: ["DREDGEMVPApp"]
        )
    ],
    dependencies: [
        .package(path: "DREDGE")
    ],
    targets: [
        .executableTarget(
            name: "DREDGECli",
            dependencies: [
                .product(name: "DREDGE", package: "DREDGE")
            ],
            path: "Sources"
        ),
        .target(
            name: "DREDGEMVPApp",
            dependencies: [
                .product(name: "DREDGE", package: "DREDGE")
            ],
            path: "DREDGE_MVP_App",
            resources: [
                .process("AboutStrings.strings")
            ]
        ),
        .testTarget(
            name: "DREDGE-CliTests",
            dependencies: ["DREDGECli"],
            path: "Tests/DREDGE-CliTests"
        )
    ]
)
