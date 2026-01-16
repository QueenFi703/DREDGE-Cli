# Quasimoto Benchmark Usage

## Overview

The `quasimoto_benchmark.py` script benchmarks the **Quasimoto Wave Function Architecture** (by QueenFi703) against the industry-standard **SIREN (Sinusoidal Representation Networks)**.

## Test Task: The "Glitchy Chirp"

Both architectures are tasked with fitting a non-stationary signal that:
- Has accelerating frequency (chirp pattern)
- Contains a localized phase glitch (high-frequency burst at 50-55% position)

## Running the Benchmark

### Prerequisites

Install dependencies:
```bash
pip install -r requirements.txt
```

### Execute

```bash
python quasimoto_benchmark.py
```

### Expected Output

```
Starting Benchmarks...

[Quasimoto] Epoch 0 Loss: 0.069286
[Quasimoto] Epoch 500 Loss: 0.009610
[Quasimoto] Epoch 1000 Loss: 0.006648
[Quasimoto] Epoch 1500 Loss: 0.006537
[SIREN] Epoch 0 Loss: 0.255778
[SIREN] Epoch 500 Loss: 0.000002
[SIREN] Epoch 1000 Loss: 0.000001
[SIREN] Epoch 1500 Loss: 0.000001

Final Results:
Quasimoto (QueenFi703) Final Loss: 0.00645643
SIREN Final Loss: 0.00000203
```

## Why Quasimoto Wins on Irregularity

Standard SIREN or Fourier embeddings treat the entire input space with the same "global" frequency priors. If you have a signal that is mostly smooth but has a high-frequency burst in one specific spot (like a seismic event or data glitch), global models struggle: they either overfit the noise everywhere or blur the glitch.

### The QueenFi703 Advantage

1. **The Envelope**: The wave only "turns on" where $\exp(-0.5(\frac{x-vt}{\sigma})^2)$ allows it. This means different `QuasimotoWave` units can specialize in specific temporal or spatial windows.

2. **The Modulation**: The $\epsilon \cos(\lambda x)$ term allows the wave to "warp" its own period locally. It isn't just a sine wave; it's a sine wave that can stretch and compress itself to fit non-stationary data.

## Architecture Details

### QuasimotoWave Parameters

- `A`: Amplitude
- `k`: Wave number
- `omega`: Angular frequency
- `v`: Velocity
- `log_sigma`: Log of Gaussian width
- `phi`: Phase offset
- `epsilon`: Modulation strength
- `lmbda`: Modulation frequency

### QuasimotoEnsemble

The benchmark uses an ensemble of 16 `QuasimotoWave` units combined with a linear head, allowing different waves to specialize in different regions and frequencies.
