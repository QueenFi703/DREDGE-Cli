# Docker Workflow Methodology

## Overview

This document describes the **File Sorter Cadence Methodology** applied to Docker CI/CD workflows in this repository. This methodology organizes workflow steps into logical phases with appropriate timing controls to optimize build performance, disk space management, and reliability.

## Methodology Principles

### 1. **Phase-Based Organization (Sorting Algorithm)**

Workflow steps are organized into distinct phases, executed in a specific order for optimal resource usage:

#### Phase 1: Repository Setup (Initial)
- **Purpose**: Prepare the workspace
- **Actions**: Checkout repository
- **Timing**: Immediate start

#### Phase 2: Environment Preparation (Pre-build)
- **Purpose**: Set up tools and free resources
- **Actions**: 
  - Configure Docker Buildx/QEMU
  - Execute disk cleanup (sorted by size impact)
- **Sorting**: Removals ordered from largest to smallest impact
  1. Android SDK (~8GB)
  2. CodeQL (~6GB)
  3. .NET SDK (~2GB)
  4. GHC/Haskell (~1.5GB)
  5. Docker system prune
- **Timing**: Before any builds

#### Phase 3: Registry Authentication (Pre-build)
- **Purpose**: Authenticate with container registry
- **Actions**: Login to GHCR
- **Timing**: After environment prep, before builds

#### Phase 4: Build Execution (Main)
- **Purpose**: Build and push Docker images
- **Actions**: Docker build with BuildKit caching
- **Timing**: Core execution phase

#### Phase 5: Attestation and Security (Post-build)
- **Purpose**: Security validation and provenance
- **Actions**:
  - Generate attestations
  - Run Trivy scanner
  - Upload security reports
- **Timing**: After successful builds

#### Phase 6: Cleanup (Terminal)
- **Purpose**: Free resources for subsequent jobs
- **Actions**:
  - Remove test containers/images
  - Prune build cache (older than 1 hour)
  - Report remaining disk space
- **Timing**: Always runs (`if: always()`)

### 2. **Cadence Controls (Timing/Rhythm)**

Each operation has appropriate wait times based on complexity:

#### Test Image Cadence
- **CPU images**: 15 second warmup
  - Simpler initialization
  - Fast service startup
  
- **GPU images**: 20 second warmup
  - CUDA library loading required
  - More complex initialization
  
- **Post-build tests**: Variable timing (10-15s)
  - GPU builds: 15s for CUDA initialization
  - CPU builds: 10s for service startup

#### Cleanup Cadence
- **Build cache pruning**: Filter `until=1h`
  - Preserves recent cache for reuse
  - Removes stale artifacts
  
- **Always execute**: `if: always()`
  - Ensures cleanup even on failures
  - Prevents disk accumulation

### 3. **File Operation Organization**

Disk operations follow a specific order:

1. **Pre-build cleanup**: Maximize available space
2. **Build operations**: Use freed space efficiently
3. **Test operations**: Minimal disk footprint
4. **Post-build cleanup**: Prepare for next job

## Applied Workflows

### docker-test.yml
- **Purpose**: Validate Docker builds on PRs
- **Matrix**: cpu-build, gpu-build, dev
- **Phases**: All 6 phases implemented
- **Key optimization**: Post-test cleanup prevents accumulation

### docker-publish.yml
- **Purpose**: Build and publish to GHCR
- **Matrix**: cpu-build, gpu-build, dev
- **Phases**: All 6 phases implemented
- **Key optimization**: 
  - Post-build cleanup after attestation
  - Cadenced testing in separate job

## Benefits

1. **Disk Space Management**: 
   - Pre-build cleanup frees ~17GB
   - Post-cleanup prevents accumulation
   - Sorted by impact for maximum recovery

2. **Reliability**:
   - Phased approach isolates failures
   - Always-cleanup ensures consistency
   - Cadenced waits prevent race conditions

3. **Performance**:
   - Optimal step ordering
   - Appropriate timing prevents unnecessary waits
   - Cache strategy balances space vs. speed

4. **Maintainability**:
   - Clear phase comments
   - Consistent structure across workflows
   - Self-documenting timing rationale

## Disk Space Tracking

Each phase reports disk usage:

```bash
# Before cleanup
df -h | grep -E '(Filesystem|/dev/root)'

# After operations
df -h | grep -E '(Filesystem|/dev/root)'
```

This provides visibility into space consumption patterns.

## Future Enhancements

- [ ] Add phase timing metrics collection
- [ ] Implement adaptive cadence based on runner load
- [ ] Add disk space threshold alerts
- [ ] Optimize cache retention policies

## References

- **Original issue**: GPU builds failing with "no space left on device"
- **Root cause**: Large CUDA libraries (~84GB) on 84GB runners
- **Solution**: Phase-based cleanup methodology with cadence controls
