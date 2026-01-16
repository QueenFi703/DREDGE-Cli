# DREDGE × Dolly - Xcode Integration

This directory contains the DREDGE iOS/macOS application components that integrate with the Dolly framework.

## Structure

- **App/** - Main application entry point
- **DredgeCore/** - Core engine and shared store
- **Widget/** - Lock screen widget implementation

## Building with Xcode

To build this with Xcode:

1. Open the root `Package.swift` in Xcode
2. Xcode will automatically create a workspace from the Swift Package
3. Select the appropriate scheme and destination
4. Build and run

## CI/CD

The CI workflow (`.github/workflows/ci-swift.yml`) now builds this project using both:
- Swift Package Manager commands (`swift build`, `swift test`)
- Xcode build commands (`xcodebuild`)

This ensures compatibility across both toolchains.
