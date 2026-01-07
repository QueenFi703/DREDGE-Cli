# µH-iOS Integration with DREDGE

This document describes how µH-iOS integrates with DREDGE to provide a formally verified compute platform.

## Overview

µH-iOS serves as a **root of trust** for the DREDGE orchestration system, providing:
- Formally verified isolation between guest workloads
- Memory non-interference guarantees
- Capability-based access control
- Deterministic VM exit handling

DREDGE uses µH-iOS to:
- Run isolated compute workloads
- Orchestrate guest VMs
- Enforce security policies
- Manage resource allocation

## Architecture

```
┌─────────────────────────────────────────────────────┐
│               DREDGE Orchestration                  │
│  (Policy Definition, Workload Management, Dolly)    │
└─────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────┐
│              µH-iOS Verified Core                   │
│   (Memory Isolation, Capability Enforcement)        │
└─────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────┐
│          Apple Hypervisor.framework (HVF)           │
└─────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────┐
│                   XNU Kernel                        │
└─────────────────────────────────────────────────────┘
```

## Integration Points

### 1. Guest Workload Execution

DREDGE can launch isolated compute tasks within µH-iOS VMs:

```swift
import UHiOSApp

// Create VM via µH-iOS
let config = UHiOSCore.VMConfig(
    memorySize: 512 * 1024 * 1024,
    vcpuCount: 1,
    name: "DREDGE Worker"
)

let vm = try UHiOSCore.shared.createVM(config: config)
try UHiOSCore.shared.initializeVM(vm, entryPoint: 0x1000)

// Run DREDGE workload in isolated VM
let exitReason = try UHiOSCore.shared.runVM(vm)
```

### 2. Dolly Integration

Dolly can lift compute tasks between CPU and GPU contexts within the safety boundaries provided by µH-iOS:

```python
from dredge import lift_insight
from uh_ios import VMContext

# Create isolated compute context
with VMContext() as vm:
    # Lift insight using Dolly within µH-iOS isolation
    result = lift_insight("Digital memory must be human-reachable.", vm=vm)
    print(f"Lifted in isolated VM: {result['id']}")
```

### 3. Formal Guarantees

When DREDGE uses µH-iOS, it benefits from formal properties:

1. **Memory Isolation**: DREDGE workloads cannot access each other's memory
2. **Capability Soundness**: Only explicitly granted operations are allowed
3. **Determinism**: Same inputs produce same outputs (reproducible results)
4. **Totality**: All VM exits are handled safely

## Use Cases

### Secure ML Inference

Run machine learning models in isolated VMs with formally verified memory isolation:

```swift
// Create isolated VM for ML inference
let mlVM = try UHiOSCore.shared.createVM(
    config: .init(memorySize: 1024 * 1024 * 1024, name: "ML Inference")
)

// Load model and data (isolated)
// Run inference
// Extract results
```

### Confidential Data Processing

Process sensitive data within formally verified isolation boundaries:

```swift
// Create isolated VM for confidential processing
let confidentialVM = try UHiOSCore.shared.createVM(
    config: .init(memorySize: 512 * 1024 * 1024, name: "Confidential")
)

// Process confidential data in isolation
// Memory isolation formally verified
```

### Multi-Tenant Computation

Run untrusted code from multiple sources with guaranteed isolation:

```swift
// Create VMs for different tenants
let tenant1VM = try UHiOSCore.shared.createVM(config: .init(name: "Tenant 1"))
let tenant2VM = try UHiOSCore.shared.createVM(config: .init(name: "Tenant 2"))

// Memory non-interference formally guaranteed
// ∀ vm₁ vm₂. vm₁ ≠ vm₂ → memory(vm₁) ∩ memory(vm₂) = ∅
```

## Security Benefits

### For DREDGE

1. **Formal Verification**: Mathematical proofs of isolation properties
2. **Minimal TCB**: Small, auditable trusted code base (~3500 LOC)
3. **Platform Compliance**: Works on stock iOS without jailbreaks
4. **Memory Safety**: Rust implementation prevents memory corruption

### For Workloads

1. **Strong Isolation**: Hardware-assisted virtualization + formal proofs
2. **Controlled Sharing**: Explicit capability-based access control
3. **Deterministic Behavior**: Reproducible execution
4. **Auditable**: All state transitions formally specified

## Building the Integration

### Prerequisites

- Rust 1.70+
- Swift 5.9+
- iOS 16+ or macOS 13+ (for HVF support)
- Xcode (for iOS development)

### Build Steps

```bash
# Build µH-iOS Rust core
cd uh-ios
cargo build --release
cargo test

# Build Swift application
cd ../uh-ios-app
swift build
swift test

# Run DREDGE with µH-iOS support
cd ..
python -m dredge serve --with-uhios
```

## API Reference

### Rust Core API

See `uh-ios/README.md` for detailed Rust API documentation.

### Swift API

See `uh-ios-app/Sources/UHiOSApp/UHiOSCore.swift` for Swift API documentation.

### Python Bindings (Planned)

Python bindings for DREDGE integration are planned for future releases:

```python
import uh_ios

# Create VM
vm = uh_ios.create_vm(memory_mb=512, vcpus=1)

# Run workload
result = vm.run_workload(workload_fn)

# Clean up
vm.halt()
```

## Testing

### Unit Tests

```bash
# Test Rust core
cd uh-ios
cargo test

# Test Swift layer
cd ../uh-ios-app
swift test
```

### Integration Tests

```bash
# Test DREDGE + µH-iOS integration
python -m pytest tests/test_uhios_integration.py
```

### Property-Based Tests

The Rust core includes property-based tests using proptest:

```bash
cd uh-ios
cargo test --features proptest
```

## Performance

µH-iOS prioritizes correctness over performance, but provides reasonable throughput:

- VM creation: ~100ms
- VM initialization: ~50ms
- Context switch: ~10µs (dependent on HVF)
- Memory mapping: ~1µs per page

For high-performance workloads, consider:
- Batching operations
- Reusing VMs
- Pre-mapping memory

## Limitations

Current limitations:
1. Single-threaded execution
2. No device passthrough
3. Limited to iOS 16+ / macOS 13+
4. Requires Hypervisor.framework entitlement

Future work will address these limitations.

## Contributing

See the main DREDGE repository for contribution guidelines:
https://github.com/QueenFi703/DREDGE

## License

MIT License - See LICENSE file

## References

- µH-iOS Paper: `docs/uh-ios-paper.md`
- µH-iOS Core: `uh-ios/README.md`
- DREDGE Documentation: `README.md`
- Dolly Integration: `DollyIntegration.md`
