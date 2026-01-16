# Swift Development Guide

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
    // CLI executable (from swift/Sources/main.swift)
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
    // CLI executable (from Sources/main.swift)
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

- **Swift CLI Version:** 0.1.0 (as defined in Sources/main.swift)
- **Python CLI Version:** 0.1.4 (as defined in pyproject.toml)
- **Swift Tools Version:** 5.9+

## Troubleshooting

### Build Fails from Root
- Ensure you're in the repository root directory
- Check that `swift/Sources/main.swift` exists
- Verify Swift toolchain is installed: `swift --version`

### Build Fails from swift/
- Ensure you're in the `swift/` subdirectory
- Check that `Sources/main.swift` exists
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
