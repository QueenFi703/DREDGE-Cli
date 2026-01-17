# Swift Development Guide

## Quick Start

### Using Swift Package Manager (SPM)

Add DREDGE-Cli to your `Package.swift`:

```swift
// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "YourApp",
    platforms: [
        .macOS(.v12),
        .iOS(.v15)
    ],
    dependencies: [
        .package(url: "https://github.com/QueenFi703/DREDGE-Cli.git", from: "0.2.0")
    ],
    targets: [
        .executableTarget(
            name: "YourApp",
            dependencies: [
                .product(name: "DREDGECli", package: "DREDGE-Cli")
            ]
        )
    ]
)
```

Then import and use:

```swift
import DREDGECli

// String Theory calculations
let stringTheory = StringTheory(dimensions: 10)
let spectrum = stringTheory.modeSpectrum(maxModes: 10)
let mode = stringTheory.vibrationalMode(n: 1, x: 0.5)

// MCP Client for server interactions
let client = MCPClient(serverURL: "http://localhost:3002")
if #available(macOS 12.0, iOS 15.0, *) {
    let capabilities = try await client.listCapabilities()
}

// Unified DREDGE integration
let unified = UnifiedDREDGE(dimensions: 10)
if #available(macOS 12.0, iOS 15.0, *) {
    let result = try await unified.unifiedInference(
        insight: "Digital memory must be human-reachable",
        coords: [0.5, 0.5, 0.5],
        modes: [1, 2, 3]
    )
}
```

### Using Xcode

1. **Add Package Dependency:**
   - File → Add Package Dependencies...
   - Enter: `https://github.com/QueenFi703/DREDGE-Cli.git`
   - Select version: `0.2.0` or later

2. **Import and Use:**
   ```swift
   import DREDGECli
   
   let stringTheory = StringTheory()
   print(DREDGECli.version)
   ```

### Minimal Configuration

No additional configuration required! DREDGE-Cli works out of the box with sensible defaults:
- String Theory dimensions: 10 (superstring theory)
- MCP Server URL: `http://localhost:3002`
- Compatible with Swift 5.9+

## Overview

The DREDGE-Cli repository includes a root-level `Package.swift` that allows building the Swift CLI directly from the repository root. This provides a more convenient developer experience while maintaining compatibility with the self-contained Swift project in the `swift/` subdirectory.

Additionally, the repository includes an Xcode workspace (`DREDGE-Cli.xcworkspace`) for developers who prefer working with Xcode IDE.

## Structure

The repository has two Package.swift files:

### 1. Root Package.swift (`/Package.swift`)
- Located at the repository root
- References Swift source code in the `swift/` subdirectory
- Allows building and testing from the root directory
- **Target naming:** `DREDGECli` (executable)

### 2. Swift Subdirectory Package.swift (`/swift/Package.swift`)
- Located in the `swift/` subdirectory
- Self-contained Swift package configuration
- Allows building and testing from within `swift/`
- **Target naming:** `DREDGECli` (executable) - aligned with root

### 3. Xcode Workspace (`/DREDGE-Cli.xcworkspace`)
- Located at the repository root
- Contains references to both Swift packages
- Enables development using Xcode IDE
- Includes shared workspace settings

## Usage

### Using Xcode Workspace

```bash
# Open the workspace in Xcode
open DREDGE-Cli.xcworkspace
```

The workspace includes:
- Root Package.swift for building from repository root
- swift/ subdirectory for self-contained Swift package development

**Benefits:**
- Full Xcode IDE features (code completion, debugging, refactoring)
- Unified view of both Swift package configurations
- Xcode build system integration
- Source control integration

### Building from Root

```bash
# From repository root
swift build

# Run the executable
./.build/debug/dredge-cli

# Or run directly
swift run dredge-cli

# Run tests
swift test
```

### Building from swift/ Subdirectory

```bash
# From repository root
cd swift

# Build
swift build

# Run the executable
./.build/debug/dredge-cli

# Or run directly
swift run dredge-cli

# Run tests
swift test
```

## Key Features

Both Package.swift configurations:
- ✅ Produce identical executables
- ✅ Support the same test suite
- ✅ Use consistent module naming (`DREDGECli`)
- ✅ Are fully synchronized and interchangeable
- ✅ Build successfully on supported platforms

## Module Structure

### Root Package.swift Targets

```swift
targets: [
    // CLI executable (from swift/Sources/DREDGECli.swift)
    .executableTarget(
        name: "DREDGECli",
        path: "swift/Sources"
    ),
    // Test target (from swift/Tests/DREDGE-CliTests/)
    .testTarget(
        name: "DREDGE-CliTests",
        dependencies: ["DREDGECli"],
        path: "swift/Tests/DREDGE-CliTests"
    )
]
```

### Swift Subdirectory Package.swift Targets

```swift
targets: [
    // CLI executable (from Sources/DREDGECli.swift)
    .executableTarget(
        name: "DREDGECli",
        path: "Sources"
    ),
    // Test target (from Tests/DREDGE-CliTests/)
    .testTarget(
        name: "DREDGE-CliTests",
        dependencies: ["DREDGECli"],
        path: "Tests/DREDGE-CliTests"
    )
]
```

## Platform Support

Both configurations specify:
- **macOS:** 12.0+
- **iOS:** 15.0+ (platform declaration in root Package.swift)

Note: The CLI executable can build on Linux, but the iOS/macOS app components (DREDGE_MVP.swift, SharedStore.swift) require Apple platforms due to SwiftUI and other Apple framework dependencies.

## Products

Both Package.swift files produce:
- **dredge-cli** executable - Command-line tool for DREDGE

## When to Use Which?

### Use Xcode Workspace when:
- Developing with Xcode IDE
- Need advanced IDE features (debugging, refactoring, UI design)
- Prefer graphical interface for project navigation
- Want integrated build and test workflows
- Working on multiple packages simultaneously

### Use Root Package.swift when:
- Working on the entire repository (Python + Swift)
- Building all components from a single location
- Integrating Swift build into repository-wide automation
- Prefer a single build command from root
- Using command-line tools

### Use Swift Subdirectory Package.swift when:
- Focusing only on Swift development
- Want a self-contained Swift package
- Distributing the Swift package separately
- Working within the swift/ directory context

## Best Practices

1. **Keep Both Synchronized:** When modifying Swift targets or dependencies, update both Package.swift files to maintain consistency

2. **Module Naming:** Always use `DREDGECli` (no hyphen) as the module name for consistency

3. **Testing:** Run tests from both locations periodically to ensure both configurations remain compatible

4. **Documentation:** Update this guide if the package structure changes

## Version Information

- **Swift CLI Version:** 0.1.0 (as defined in Sources/DREDGECli.swift)
- **Python CLI Version:** 0.1.4 (as defined in pyproject.toml)
- **Swift Tools Version:** 5.9+

## Troubleshooting

### Build Fails from Root
- Ensure you're in the repository root directory
- Check that `swift/Sources/DREDGECli.swift` exists
- Verify Swift toolchain is installed: `swift --version`

### Build Fails from swift/
- Ensure you're in the `swift/` subdirectory
- Check that `Sources/DREDGECli.swift` exists
- Verify Swift toolchain is installed: `swift --version`

### Test Import Errors
- Ensure tests import `@testable import DREDGECli` (not `DREDGE_Cli`)
- Module name changed from `DREDGE-Cli` to `DREDGECli` for consistency

### Different Build Results
- This should not happen - both configurations should produce identical executables
- If you see differences, file an issue as both Package.swift files should be synchronized

## Maintenance

This configuration was established on 2026-01-16 as part of the repository structure validation and cleanup. Both Package.swift files are maintained and should remain synchronized.

For questions or issues with the Swift build configuration, refer to the Swift Package Manager documentation or open an issue in the repository.


## Public API Reference

### Core Types

#### `DREDGECli`

Main entry point with version information.

```swift
public struct DREDGECli {
    public static let version: String  // Current version (semver format)
    public static let tagline: String  // Project tagline
    public static func run()           // Run CLI application
}
```

**Example:**
```swift
print(DREDGECli.version)  // "0.2.0"
print(DREDGECli.tagline)  // "Digital memory must be human-reachable."
DREDGECli.run()           // Print CLI information
```

---

#### `StringTheory`

String theory calculations with vibrational modes and energy spectra.

```swift
public struct StringTheory {
    public let dimensions: Int  // Number of dimensions (default: 10)
    public let length: Double   // String length (default: 1.0)
    
    public init(dimensions: Int = 10, length: Double = 1.0)
    
    /// Calculate nth vibrational mode at position x
    public func vibrationalMode(n: Int, x: Double) -> Double
    
    /// Calculate energy level for nth mode
    public func energyLevel(n: Int) -> Double
    
    /// Generate energy spectrum
    public func modeSpectrum(maxModes: Int = 10) -> [Double]
}
```

**Example:**
```swift
let st = StringTheory(dimensions: 10, length: 1.0)
let amplitude = st.vibrationalMode(n: 1, x: 0.5)  // 1.0
let spectrum = st.modeSpectrum(maxModes: 5)
```

---

#### `MCPClient`

Client for interacting with MCP (Model Context Protocol) server.

```swift
public struct MCPClient {
    public let serverURL: String
    
    public init(serverURL: String = "http://localhost:3002")
    
    // Apple Platforms Only (macOS 12+, iOS 15+)
    @available(macOS 12.0, iOS 15.0, *)
    public func listCapabilities() async throws -> [String: Any]
    
    @available(macOS 12.0, iOS 15.0, *)
    public func sendRequest(operation: String, params: [String: Any]) async throws -> [String: Any]
}
```

---

#### `UnifiedDREDGE`

Unified interface combining String Theory and MCP Client.

```swift
public struct UnifiedDREDGE {
    public let stringTheory: StringTheory
    public let mcpClient: MCPClient
    
    public init(dimensions: Int = 10, serverURL: String = "http://localhost:3002")
    
    @available(macOS 12.0, iOS 15.0, *)
    public func unifiedInference(insight: String, coords: [Double], modes: [Int]) async throws -> [String: Any]
    
    @available(macOS 12.0, iOS 15.0, *)
    public func getStringSpectrum(maxModes: Int = 10) async throws -> [String: Any]
}
```

---

#### `MCPError`

Error type for MCP operations.

```swift
public enum MCPError: Error {
    case invalidURL      // Malformed server URL
    case invalidResponse // Server returned non-JSON response
    case networkError    // Network connection failed
}
```

---

### Platform Support

**All Platforms:**
- `DREDGECli`, `StringTheory` - Work everywhere
- `MCPClient`, `UnifiedDREDGE` - Initialization works everywhere

**Apple Platforms Only (macOS 12+, iOS 15+):**
- Network operations (async methods)
- Fallback methods provided for other platforms

---

### Semantic Versioning

DREDGE-Cli follows [semantic versioning](https://semver.org/):

**Current Version:** `0.2.0`

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

---

### Testing

```bash
# Run all tests
swift test

# Run specific suite
swift test --filter IntegrationTests
swift test --filter EndToEndTests
```

**Note:** End-to-end tests require running MCP server:
```bash
python -m dredge mcp  # Terminal 1
swift test            # Terminal 2
```

---

### Further Reading

- [Troubleshooting Guide](./docs/TROUBLESHOOTING.md)
- [README](./README.md)
- [MCP Server Documentation](./README.md#mcp-server-port-3002---quasimoto-integration)

---

*API Reference last updated: 2026-01-17*
