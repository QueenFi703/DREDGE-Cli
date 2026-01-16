# Quasimoto Experimentation Guide

## Overview

This guide provides ideas and code examples for extending the Quasimoto benchmark beyond the basic chirp signal experiment.

## Experiment Ideas

### 1. Multi-Dimensional Latent Spaces (Quasimoto Manifolds)

Extend QuasimotoWave to handle 2D or 3D inputs for image or volumetric data.

```python
class QuasimotoWave2D(nn.Module):
    """2D extension of QuasimotoWave for image data"""
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(1.0))
        self.kx = nn.Parameter(torch.randn(()))
        self.ky = nn.Parameter(torch.randn(()))
        self.omega = nn.Parameter(torch.randn(()))
        self.vx = nn.Parameter(torch.randn(()))
        self.vy = nn.Parameter(torch.randn(()))
        self.log_sigma = nn.Parameter(torch.zeros(()))
        self.phi = nn.Parameter(torch.zeros(()))
        self.epsilon = nn.Parameter(torch.tensor(0.1))
        self.lmbda_x = nn.Parameter(torch.randn(()))
        self.lmbda_y = nn.Parameter(torch.randn(()))

    def forward(self, x, y, t):
        sigma = torch.exp(self.log_sigma)
        phase = self.kx * x + self.ky * y - self.omega * t
        dx = x - self.vx * t
        dy = y - self.vy * t
        envelope = torch.exp(-0.5 * ((dx**2 + dy**2) / sigma**2))
        modulation = torch.sin(self.phi + 
                              self.epsilon * torch.cos(self.lmbda_x * x + self.lmbda_y * y))
        psi_real = self.A * torch.cos(phase) * envelope * modulation
        return psi_real
```

**Use Case**: Image inpainting, texture synthesis, or fitting neural radiance fields (NeRF-style scenes).

### 2. Time-Dependent Signals (Video/Dynamic Systems)

Make time (t) non-zero to model temporal evolution.

```python
def generate_dynamic_data(num_frames=50):
    """Generate a wave packet that moves and disperses over time"""
    x = torch.linspace(-10, 10, 500).view(-1, 1)
    frames = []
    
    for frame_idx in range(num_frames):
        t = torch.full_like(x, frame_idx * 0.1)
        # Moving Gaussian pulse with dispersion
        y = torch.exp(-0.5 * ((x - 2.0 * t) / (1.0 + 0.1 * t))**2) * torch.sin(5 * x - 2 * t)
        frames.append(y)
    
    return torch.stack(frames)  # Shape: [num_frames, 500, 1]
```

**Use Case**: Video frame interpolation, fluid dynamics, or modeling physical wave propagation.

### 3. Add Random Fourier Features (RFF) Baseline

Complete the trilogy by adding RFF as mentioned in the problem statement.

```python
class RandomFourierFeatures(nn.Module):
    def __init__(self, input_dim=1, num_features=256, sigma=10.0):
        super().__init__()
        # Fixed random frequencies (not learned)
        self.register_buffer('B', torch.randn(input_dim, num_features) * sigma)
        self.linear = nn.Linear(num_features * 2, 1)  # *2 for sin and cos
        
    def forward(self, x):
        # x: [N, 1]
        projections = x @ self.B  # [N, num_features]
        features = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)
        return self.linear(features)

# Add to benchmark
rff_net = RandomFourierFeatures(input_dim=1, num_features=128, sigma=5.0)
r_loss = train_model("RFF", rff_net, x, t, y)
print(f"RFF Final Loss: {r_loss:.8f}")
```

### 4. Visualization and Analysis

Add plotting to visualize what each architecture learns.

```python
import matplotlib.pyplot as plt

def visualize_predictions(x, y_true, models, model_names):
    """Plot true signal vs model predictions"""
    fig, axes = plt.subplots(len(models) + 1, 1, figsize=(12, 8))
    
    # Plot ground truth
    axes[0].plot(x.numpy(), y_true.numpy(), 'k-', linewidth=2, label='Ground Truth')
    axes[0].set_title('Ground Truth Signal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot predictions
    for idx, (model, name) in enumerate(zip(models, model_names)):
        with torch.no_grad():
            if name == "Quasimoto":
                pred = model(x.squeeze(), torch.zeros_like(x.squeeze())).view(-1, 1)
            else:
                pred = model(x)
        
        axes[idx + 1].plot(x.numpy(), y_true.numpy(), 'k--', alpha=0.3, label='True')
        axes[idx + 1].plot(x.numpy(), pred.numpy(), 'r-', label=f'{name} Prediction')
        axes[idx + 1].set_title(f'{name} Fit')
        axes[idx + 1].legend()
        axes[idx + 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quasimoto_comparison.png', dpi=150)
    print("Saved visualization to quasimoto_comparison.png")

# Use after training
visualize_predictions(x, y, [quasimoto_net, siren_net], ["Quasimoto", "SIREN"])
```

### 5. Different Signal Types

Test on various challenging signals:

```python
# Discontinuous step function
def generate_step_data():
    x = torch.linspace(-10, 10, 1000).view(-1, 1)
    t = torch.zeros_like(x)
    y = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
    return x, t, y

# Multi-frequency composition
def generate_multifreq_data():
    x = torch.linspace(-10, 10, 1000).view(-1, 1)
    t = torch.zeros_like(x)
    y = torch.sin(x) + 0.5 * torch.sin(5 * x) + 0.25 * torch.sin(20 * x)
    return x, t, y

# Noisy signal
def generate_noisy_data(noise_level=0.1):
    x = torch.linspace(-10, 10, 1000).view(-1, 1)
    t = torch.zeros_like(x)
    y = torch.sin(2 * x) * torch.exp(-0.1 * x**2)
    y += noise_level * torch.randn_like(y)
    return x, t, y
```

### 6. Hyperparameter Tuning

Experiment with architecture parameters:

```python
# Test different ensemble sizes
for n_waves in [4, 8, 16, 32]:
    model = QuasimotoEnsemble(n=n_waves)
    loss = train_model(f"Quasimoto-{n_waves}", model, x, t, y, epochs=1000)
    print(f"Ensemble size {n_waves}: Final Loss = {loss:.8f}")

# Test different learning rates
for lr in [1e-4, 1e-3, 1e-2]:
    model = QuasimotoEnsemble(n=16)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # ... train with this optimizer
```

### 7. Transfer Learning / Fine-tuning

Pre-train on simple signals, then fine-tune on complex ones:

```python
# Phase 1: Pre-train on simple sine wave
x_simple = torch.linspace(-10, 10, 1000).view(-1, 1)
y_simple = torch.sin(x_simple)
t_simple = torch.zeros_like(x_simple)

model = QuasimotoEnsemble(n=16)
train_model("Quasimoto-Pretrain", model, x_simple, t_simple, y_simple, epochs=1000)

# Phase 2: Fine-tune on complex chirp
x_complex, t_complex, y_complex = generate_data()
train_model("Quasimoto-Finetune", model, x_complex, t_complex, y_complex, epochs=1000)
```

### 8. Attention Mechanisms

Add learnable attention to ensemble waves:

```python
class QuasimotoEnsembleWithAttention(nn.Module):
    def __init__(self, n=16):
        super().__init__()
        self.waves = nn.ModuleList([QuasimotoWave() for _ in range(n)])
        self.attention = nn.Sequential(
            nn.Linear(n, n),
            nn.Softmax(dim=-1)
        )
        self.head = nn.Linear(n, 1)
    
    def forward(self, x, t):
        feats = torch.stack([w(x, t) for w in self.waves], dim=-1)
        attn_weights = self.attention(feats)
        weighted_feats = feats * attn_weights
        return self.head(weighted_feats)
```

## Performance Metrics to Track

Beyond MSE loss, consider:

1. **Peak Signal-to-Noise Ratio (PSNR)**
   ```python
   def psnr(pred, target):
       mse = torch.mean((pred - target) ** 2)
       return 20 * torch.log10(1.0 / torch.sqrt(mse))
   ```

2. **Structural Similarity Index (SSIM)** - for 2D signals

3. **Convergence Speed** - epochs to reach target loss

4. **Parameter Efficiency** - performance per parameter count
   ```python
   num_params = sum(p.numel() for p in model.parameters())
   print(f"Parameters: {num_params:,}")
   ```

## Publication and Sharing

Consider writing up results as:
- Technical blog post
- arXiv paper
- Kaggle notebook
- GitHub repository with examples

## Next Steps

1. Start with visualization (#4) to understand current behavior
2. Add RFF baseline (#3) for complete comparison
3. Experiment with 2D images (#1) for broader impact
4. Document findings in a research note or blog post
