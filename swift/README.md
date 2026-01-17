# Swift Implementation

This directory contains Swift-related files for the DREDGE project.

## Contents

- **Package.swift** - Swift Package Manager configuration
- **Sources/** - CLI executable source code (main.swift)
- **Tests/** - Test files
- **DREDGE_MVP_App/** - iOS MVP app implementation
  - DREDGE_MVP.swift - iOS app with SwiftUI
  - SharedStore.swift - Shared store implementation
  - AboutStrings.strings - Localized strings

## Building

To build the Swift CLI:

```bash
cd swift
swift build
```

To run:

```bash
swift run dredge-cli
```
