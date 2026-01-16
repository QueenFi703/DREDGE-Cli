# About NanoGPT (124M) — DREDGE Core

## Overview

**NanoGPT (124M)** is the cognitive core of **DREDGE**.

Rather than scaling parameters, DREDGE scales *intent*.  
NanoGPT operates as a **glass-box transformer**—compact, inspectable, and deployable close to the data.

Within the DREDGE architecture, NanoGPT works in concert with:

- **train_switch** — dynamically routes training and inference modes, enabling controlled adaptation without drift.
- **Dolly** — the GPU↔CPU lifter and execution steward, ensuring efficient, portable inference.
- **DREDGE Agents** — edge-deployed listeners that collect raw signal and return surfaced insight.

## What NanoGPT Does in DREDGE

NanoGPT is responsible for:

- Attention-driven signal extraction (“dredging”)
- Lightweight memory shaping across sessions
- Deterministic, auditable reasoning paths
- On-device or near-edge inference

This model is not a general-purpose oracle.  
It is a precision instrument.

Small model. Clear mind.

## Design Philosophy

DREDGE favors:
- Insight density over parameter count
- Decentralized execution over centralized scale
- Control, auditability, and intent alignment

NanoGPT (124M) enables rapid iteration, fine-tuning, and deployment without infrastructure gravity.

Train it lightly. Guide it carefully.  
Let it dredge what matters.
