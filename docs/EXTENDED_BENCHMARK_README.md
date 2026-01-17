# Quasimoto Extended Benchmark

## Overview

This extended benchmark adds three major enhancements to the original Quasimoto vs SIREN comparison:

1. **Random Fourier Features (RFF) Baseline** - Completes the architecture comparison trilogy
2. **Visualization Tools** - Generates publication-quality plots of predictions and convergence
3. **4D Extension (QuasimotoWave4D)** - Spatiotemporal volumetric data support

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Extended Benchmark

```bash
python quasimoto_extended_benchmark.py
```

This will:
- Train Quasimoto, SIREN, and RFF on the 1D glitchy chirp signal (2000 epochs)
- Generate comparison visualizations
- Train Quasimoto-4D on spatiotemporal 3D volumetric data (1000 epochs)
- Save all plots as PNG files

## Output Files

The benchmark generates three visualization files:

- **`quasimoto_comparison.png`** - Side-by-side comparison of all three architectures' predictions
- **`quasimoto_convergence.png`** - Training convergence curves on log scale
- **`quasimoto_4d_convergence.png`** - 4D model training convergence

## Architecture Details

### 1. Random Fourier Features (RFF)

```python
class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim=1, num_features=256, sigma=10.0):
        super().__init__()
        # Fixed random frequencies (not learned)
        self.register_buffer('B', torch.randn(input_dim, num_features) * sigma)
        self.linear = nn.Linear(num_features * 2, 1)
```

**Key Characteristics:**
- Uses **fixed** random frequencies (not learned)
- Projects input to high-dimensional feature space
- Combines sin and cos features for universal approximation
- Simple linear layer on top of features

**Advantages:**
- Fast to train (fewer parameters to learn)
- Good for smooth functions

**Disadvantages:**
- Cannot adapt frequency distribution to data
- Struggles with localized irregularities

### 2. QuasimotoWave4D - Spatiotemporal Extension

```python
class QuasimotoWave4D(nn.Module):
    def forward(self, x, y, z, t):
        phase = self.kx * x + self.ky * y + self.kz * z - self.omega * t
        dx, dy, dz = x - self.vx * t, y - self.vy * t, z - self.vz * t
        envelope = torch.exp(-0.5 * ((dx**2 + dy**2 + dz**2) / sigma**2))
        modulation = torch.sin(self.phi + 
                              self.epsilon * torch.cos(self.lmbda_x * x + 
                                                       self.lmbda_y * y + 
                                                       self.lmbda_z * z))
        return self.A * torch.cos(phase) * envelope * modulation
```

**Parameters (per wave):**
- `kx, ky, kz`: Wave numbers for each spatial dimension
- `vx, vy, vz`: Velocities for each spatial dimension
- `lmbda_x, lmbda_y, lmbda_z`: Modulation frequencies for 3D space
- Plus standard `A, omega, sigma, phi, epsilon`

**Use Cases:**
- **Medical Imaging**: 4D CT/MRI (3D space + time)
- **Fluid Dynamics**: Spatiotemporal flow fields
- **Weather Modeling**: Atmospheric phenomena evolution
- **Video Processing**: 3D volumetric video data

### 3. Visualization Functions

Two main visualization functions are provided:

#### `visualize_predictions(x, y_true, models, model_names)`
Creates a multi-panel plot showing:
- Ground truth signal
- Each model's prediction with MSE
- Residuals visible in overlay

#### `visualize_convergence(losses_dict)`
Generates log-scale convergence plot showing:
- Training loss over epochs for all models
- Comparative convergence rates
- Final loss values

## Results Interpretation

### 1D Benchmark Results (Typical)

| Architecture | Final Loss | Convergence Speed | Glitch Handling |
|-------------|------------|-------------------|-----------------|
| **SIREN** | ~10^-5 | Very Fast | Excellent |
| **RFF** | ~10^-3 | Fast | Good |
| **Quasimoto** | ~10^-2 | Moderate | Specialized |

**Key Insights:**

1. **SIREN** achieves lowest loss due to global optimization across entire signal
2. **RFF** converges quickly but plateaus due to fixed frequencies
3. **Quasimoto** shows regional specialization - different waves focus on different areas

### 4D Benchmark Results

The 4D extension demonstrates Quasimoto's scalability:
- **Grid Size**: 20x20x20 = 8,000 points
- **Training**: ~1000 epochs
- **Final Loss**: Typically < 10^-3
- **Key Feature**: Each wave can localize in 3D space and track temporal evolution

## Customization

### Adjust Training Duration

```python
# In quasimoto_extended_benchmark.py, modify:
train_model(name, model, x, t, y, epochs=5000)  # Increase for better convergence
```

### Change RFF Parameters

```python
# More features = better approximation
rff_net = RandomFourierFeatures(input_dim=1, num_features=512, sigma=10.0)
```

### Modify 4D Grid Resolution

```python
# Larger grid = more detailed but slower training
X, Y, Z, T, signal = generate_4d_data(grid_size=30)  # 27,000 points
```

### Custom Ensemble Sizes

```python
quasimoto_net = QuasimotoEnsemble(n=32)  # More waves = more capacity
quasimoto_4d = QuasimotoEnsemble4D(n=16)  # Adjust for 4D complexity
```

## Performance Comparison

### Parameter Counts

| Model | Architecture | Parameters |
|-------|-------------|------------|
| Quasimoto (n=16) | 16 waves + linear | ~8 × 16 + 16 = 144 |
| SIREN (64-64) | 3 layers | ~1 + 64×64 + 64 = 4,161 |
| RFF (128 features) | Random + linear | 128×2 = 256 (learned) |

### Computational Cost

- **SIREN**: Highest (multiple matrix multiplications)
- **Quasimoto**: Moderate (parallel wave computation)
- **RFF**: Lowest (single matrix multiply after projection)

## Extending Further

### Add More Baselines

```python
# Fourier Neural Operator
# Neural ODE
# Implicit Neural Representations (INR)
```

### Try Different Signals

```python
def generate_multifreq_data():
    x = torch.linspace(-10, 10, 1000).view(-1, 1)
    y = torch.sin(x) + 0.5*torch.sin(5*x) + 0.25*torch.sin(20*x)
    return x, torch.zeros_like(x), y
```

### 3D Visualization for 4D Data

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Extract predictions at specific time
with torch.no_grad():
    pred = model(X, Y, Z, T).reshape(grid_size, grid_size, grid_size)

# Plot slices or isosurfaces
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
# ... 3D plotting code
```

## Research Applications

### 1. Seismic Signal Processing
- Detect earthquakes in specific spatial regions
- Model wave propagation with localized sources

### 2. Medical Imaging
- 4D cardiac MRI (3D heart + time)
- Tumor tracking in dynamic CT scans

### 3. Financial Time Series
- Detect anomalies in specific time windows
- Model regime changes in market data

### 4. Video Compression
- Represent video as continuous 4D function
- Adaptive resolution in different regions

## Citation

If you use this benchmark in your research, please cite:

```
@software{quasimoto2024,
  author = {QueenFi703},
  title = {Quasimoto: Learnable Wave Functions for Non-Stationary Signal Processing},
  year = {2024},
  url = {https://github.com/QueenFi703/DREDGE}
}
```

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- **SIREN**: Implicit Neural Representations with Periodic Activation Functions (Sitzmann et al., 2020)
- **Random Fourier Features**: Random Features for Large-Scale Kernel Machines (Rahimi & Recht, 2007)
- **Quasimoto Architecture**: Original work by QueenFi703 (2026)
