# Mobile Optimization (Design Intent)

## Definition
Ensure DREDGE-Cli runs efficiently, legibly, and reliably inside mobile terminal environments (Termux/iSH/SSH-on-mobile), with constrained resources and small-screen ergonomics as first-class concerns.

## Supported Environments
- Android (Termux) — primary target, real Linux userland, predictable behavior.
- iOS (iSH + SSH) — Alpine quirks, stricter syscalls; tested subset.
- Mobile SSH sessions — inherits mobile TTY width/latency constraints.

## Defaults vs Overrides
- CPU/Threads: default to 1 thread; no multiprocessing unless explicitly requested.
- Memory: favor streaming; avoid large in-memory buffers; chunked reads.
- Network: minimize round trips; smaller payloads; reuse connections.
- Output: narrow-width aware; concise help; progress/spinners by default; `--verbose` for full logs.
- Caching: prefer on-device cache dirs (e.g., Termux: `$HOME/.cache/dredge`); avoid oversized artifacts by default.

## Environment Detection
- Termux: `TERMUX_VERSION` present.
- iSH heuristic: `uname -a` shows Alpine or contains `ish`; fallback to POSIX + reduced features.
- SSH/mobile TTY: detect terminal width; fall back to 80 or user-provided.

## Non-Goals
- No touch UI in this phase.
- No browser/mobile-web UI (Phase II).
- No GPU expectations on-device; assume CPU-only.
- No heavy background daemons by default.

## Overrides (opt-in)
- `--threads N` to raise concurrency.
- `--no-spinner` for CI/log pipes.
- `--cache-dir PATH` to relocate caches.
- `--verbose` for full logs; `--quiet` for minimal.
