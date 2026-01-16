import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math

# --- CREDITS ---
# Quasimoto Wave Function Architecture by: QueenFi703
# Extended with RFF baseline, visualization, 4D/6D support, and Interference Basis
# ----------------

class QuasimotoWave(nn.Module):
    """
    Author: QueenFi703
    Learnable continuous latent wave representation with controlled phase irregularity.
    """
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(1.0))
        self.k = nn.Parameter(torch.randn(()))
        self.omega = nn.Parameter(torch.randn(()))
        self.v = nn.Parameter(torch.randn(()))
        self.log_sigma = nn.Parameter(torch.zeros(()))
        self.phi = nn.Parameter(torch.zeros(()))
        self.epsilon = nn.Parameter(torch.tensor(0.1))
        self.lmbda = nn.Parameter(torch.randn(()))

    def forward(self, x, t):
        sigma = torch.exp(self.log_sigma)
        phase = self.k * x - self.omega * t
        envelope = torch.exp(-0.5 * ((x - self.v * t) / sigma) ** 2)
        modulation = torch.sin(self.phi + self.epsilon * torch.cos(self.lmbda * x))
        
        # Real-only version for standard MSE benchmarking
        psi_real = self.A * torch.cos(phase) * envelope * modulation
        return psi_real

class QuasimotoWave4D(nn.Module):
    """
    Author: QueenFi703
    4D extension of QuasimotoWave for spatiotemporal volumetric data (x, y, z, t).
    Use cases: Medical imaging (4D CT/MRI), fluid dynamics, weather modeling.
    """
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(1.0))
        # Wave numbers for each spatial dimension
        self.kx = nn.Parameter(torch.randn(()))
        self.ky = nn.Parameter(torch.randn(()))
        self.kz = nn.Parameter(torch.randn(()))
        self.omega = nn.Parameter(torch.randn(()))
        # Velocities for each spatial dimension
        self.vx = nn.Parameter(torch.randn(()))
        self.vy = nn.Parameter(torch.randn(()))
        self.vz = nn.Parameter(torch.randn(()))
        self.log_sigma = nn.Parameter(torch.zeros(()))
        self.phi = nn.Parameter(torch.zeros(()))
        self.epsilon = nn.Parameter(torch.tensor(0.1))
        # Modulation frequencies for each spatial dimension
        self.lmbda_x = nn.Parameter(torch.randn(()))
        self.lmbda_y = nn.Parameter(torch.randn(()))
        self.lmbda_z = nn.Parameter(torch.randn(()))

    def forward(self, x, y, z, t):
        sigma = torch.exp(self.log_sigma)
        # Phase propagation in 3D space
        phase = self.kx * x + self.ky * y + self.kz * z - self.omega * t
        # Gaussian envelope centered on moving point
        dx = x - self.vx * t
        dy = y - self.vy * t
        dz = z - self.vz * t
        envelope = torch.exp(-0.5 * ((dx**2 + dy**2 + dz**2) / sigma**2))
        # 3D phase modulation
        modulation = torch.sin(self.phi + 
                              self.epsilon * torch.cos(self.lmbda_x * x + 
                                                       self.lmbda_y * y + 
                                                       self.lmbda_z * z))
        psi_real = self.A * torch.cos(phase) * envelope * modulation
        return psi_real

class QuasimotoWave6D(nn.Module):
    """
    Author: QueenFi703
    6D extension of QuasimotoWave for 5D spatial + temporal dimensions (x1, x2, x3, x4, x5, t).
    Use cases: High-dimensional physics simulations, hyperspace data, advanced quantum systems,
    multi-modal sensor fusion (e.g., RGBD + thermal + audio over time).
    """
    def __init__(self):
        super().__init__()
        self.A = nn.Parameter(torch.tensor(1.0))
        # Wave numbers for each of 5 spatial dimensions
        self.k1 = nn.Parameter(torch.randn(()))
        self.k2 = nn.Parameter(torch.randn(()))
        self.k3 = nn.Parameter(torch.randn(()))
        self.k4 = nn.Parameter(torch.randn(()))
        self.k5 = nn.Parameter(torch.randn(()))
        self.omega = nn.Parameter(torch.randn(()))
        # Velocities for each of 5 spatial dimensions
        self.v1 = nn.Parameter(torch.randn(()))
        self.v2 = nn.Parameter(torch.randn(()))
        self.v3 = nn.Parameter(torch.randn(()))
        self.v4 = nn.Parameter(torch.randn(()))
        self.v5 = nn.Parameter(torch.randn(()))
        self.log_sigma = nn.Parameter(torch.zeros(()))
        self.phi = nn.Parameter(torch.zeros(()))
        self.epsilon = nn.Parameter(torch.tensor(0.1))
        # Modulation frequencies for each of 5 spatial dimensions
        self.lmbda_1 = nn.Parameter(torch.randn(()))
        self.lmbda_2 = nn.Parameter(torch.randn(()))
        self.lmbda_3 = nn.Parameter(torch.randn(()))
        self.lmbda_4 = nn.Parameter(torch.randn(()))
        self.lmbda_5 = nn.Parameter(torch.randn(()))

    def forward(self, x1, x2, x3, x4, x5, t):
        sigma = torch.exp(self.log_sigma)
        # Phase propagation in 5D space
        phase = (self.k1 * x1 + self.k2 * x2 + self.k3 * x3 + 
                self.k4 * x4 + self.k5 * x5 - self.omega * t)
        # Gaussian envelope centered on moving point in 5D
        d1 = x1 - self.v1 * t
        d2 = x2 - self.v2 * t
        d3 = x3 - self.v3 * t
        d4 = x4 - self.v4 * t
        d5 = x5 - self.v5 * t
        envelope = torch.exp(-0.5 * ((d1**2 + d2**2 + d3**2 + d4**2 + d5**2) / sigma**2))
        # 5D phase modulation
        modulation = torch.sin(self.phi + 
                              self.epsilon * torch.cos(self.lmbda_1 * x1 + 
                                                       self.lmbda_2 * x2 + 
                                                       self.lmbda_3 * x3 +
                                                       self.lmbda_4 * x4 +
                                                       self.lmbda_5 * x5))
        psi_real = self.A * torch.cos(phase) * envelope * modulation
        return psi_real

class QuasimotoInterferenceBasis(nn.Module):
    """
    Author: QueenFi703
    Coupled complex-valued quasiperiodic wave basis with:
    - exp(i Σ kᵢxᵢ) interference
    - anisotropic Gaussian locality (σᵢ)
    - shared envelopes across fields
    - learned superposition weights
    """

    def __init__(self, dim, num_fields, out_features):
        super().__init__()
        self.dim = dim
        self.num_fields = num_fields
        self.out_features = out_features

        # ── Shared anisotropic envelope ─────────────────────────────
        self.log_sigma = nn.Parameter(torch.zeros(dim))
        self.v = nn.Parameter(torch.randn(dim))

        # ── Per-field wave parameters ───────────────────────────────
        self.A = nn.Parameter(torch.ones(num_fields))
        self.k = nn.Parameter(torch.randn(num_fields, dim))
        self.omega = nn.Parameter(torch.randn(num_fields))

        self.phi = nn.Parameter(torch.zeros(num_fields))
        self.epsilon = nn.Parameter(torch.full((num_fields,), 0.1))
        self.lmbda = nn.Parameter(torch.randn(num_fields, dim))

        # ── Learned superposition ───────────────────────────────────
        # Maps num_fields complex ψ → out_features real
        self.superposition = nn.Linear(2 * num_fields, out_features, bias=False)

    def forward(self, x, t):
        """
        x: [N, dim] or [N, 1]
        t: [N] or [N, 1] or scalar
        returns: [N, out_features]
        """
        # Ensure proper shapes
        if x.dim() == 2 and x.shape[1] == 1:
            # x is [N, 1], which is correct for dim=1
            pass
        elif x.dim() == 1:
            x = x.unsqueeze(-1)  # [N] → [N, 1]
        
        N = x.shape[0]
        
        if t.dim() == 0:
            t = t.expand(N)  # scalar → [N]
        elif t.dim() == 2:
            t = t.squeeze(-1)  # [N, 1] → [N]
        
        # Now x: [N, dim], t: [N]
        
        # ── Anisotropic Gaussian envelope (shared) ──────────────────
        sigma = torch.exp(self.log_sigma).clamp(min=1e-3)  # [dim]
        # Compute z for envelope: [N, dim]
        z = (x - self.v * t.unsqueeze(-1)) / sigma
        envelope_val = torch.exp(-0.5 * torch.sum(z**2, dim=-1))  # [N]

        # Optimize: Compute normalization more efficiently
        norm = torch.prod(sigma) * ((2 * math.pi) ** (self.dim / 2))
        envelope = envelope_val / norm  # [N]

        # ── exp(i Σ kᵢxᵢ − iωt) interference ─────────────────────────
        # self.k: [num_fields, dim], x: [N, dim]
        # Result: [N, num_fields]
        phase = (x @ self.k.T) - self.omega * t.unsqueeze(-1)  # [N, num_fields]
        carrier = torch.exp(1j * phase)  # [N, num_fields]

        # ── Quasiperiodic phase distortion ──────────────────────────
        # self.lmbda: [num_fields, dim]
        distortion = x @ self.lmbda.T  # [N, num_fields]
        modulation = torch.sin(self.phi + self.epsilon * torch.cos(distortion))  # [N, num_fields]

        # ── Coupled ψ fields ────────────────────────────────────────
        # Broadcast envelope [N] with other terms [N, num_fields]
        psi = self.A * carrier * envelope.unsqueeze(-1) * modulation   # [N, num_fields]

        # ── Real-valued learned superposition ───────────────────────
        psi_real = torch.cat([psi.real, psi.imag], dim=-1)  # [N, 2*num_fields]
        return self.superposition(psi_real)  # [N, out_features]

class RandomFourierFeatures(nn.Module):
    """
    Random Fourier Features (RFF) baseline.
    Uses fixed random frequencies (not learned) for feature mapping.
    """
    def __init__(self, input_dim=1, num_features=256, sigma=10.0):
        super().__init__()
        # Fixed random frequencies (not learned)
        self.register_buffer('B', torch.randn(input_dim, num_features) * sigma)
        self.linear = nn.Linear(num_features * 2, 1)  # *2 for sin and cos
        
    def forward(self, x):
        # x: [N, input_dim]
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        projections = x @ self.B  # [N, num_features]
        features = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)
        return self.linear(features)

class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30.0, is_first=False):
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        # Special initialization for SIREN
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/in_f, 1/in_f)
            else:
                self.linear.weight.uniform_(-np.sqrt(6/in_f)/w0, np.sqrt(6/in_f)/w0)

    def forward(self, x):
        return torch.sin(self.w0 * self.linear(x))

class QuasimotoEnsemble(nn.Module):
    def __init__(self, n=16):
        super().__init__()
        self.waves = nn.ModuleList([QuasimotoWave() for _ in range(n)])
        self.head = nn.Linear(n, 1)
    
    def forward(self, x, t):
        # PyTorch optimizes list comprehensions + stack efficiently
        feats = torch.stack([w(x, t) for w in self.waves], dim=-1)
        return self.head(feats)

class QuasimotoEnsemble4D(nn.Module):
    """4D ensemble for spatiotemporal data"""
    def __init__(self, n=8):
        super().__init__()
        self.waves = nn.ModuleList([QuasimotoWave4D() for _ in range(n)])
        self.head = nn.Linear(n, 1)
    
    def forward(self, x, y, z, t):
        # Optimized: Compute all waves - PyTorch can optimize this
        feats = torch.stack([w(x, y, z, t) for w in self.waves], dim=-1)
        return self.head(feats)

class QuasimotoEnsemble6D(nn.Module):
    """6D ensemble for 5D spatial + temporal data"""
    def __init__(self, n=6):
        super().__init__()
        self.waves = nn.ModuleList([QuasimotoWave6D() for _ in range(n)])
        self.head = nn.Linear(n, 1)
    
    def forward(self, x1, x2, x3, x4, x5, t):
        # Optimized: Compute all waves - PyTorch can optimize this
        feats = torch.stack([w(x1, x2, x3, x4, x5, t) for w in self.waves], dim=-1)
        return self.head(feats)

# --- BENCHMARK TASK: The "Glitchy Chirp" ---
def generate_data():
    x = torch.linspace(-10, 10, 1000).view(-1, 1)
    t = torch.zeros_like(x)
    y = torch.sin(0.5 * x**2) * torch.exp(-0.1 * x**2)
    # The Glitch - positioned at 50-55% of the signal
    glitch_start = int(0.5 * len(x))
    glitch_end = int(0.55 * len(x))
    y[glitch_start:glitch_end] += 0.5 * torch.sin(20 * x[glitch_start:glitch_end])
    return x, t, y

def generate_4d_data(grid_size=20):
    """
    Generate 4D spatiotemporal data (3D space + time).
    Optimized to reduce intermediate memory allocations.
    """
    # Create a smaller grid for 4D demo
    x = torch.linspace(-5, 5, grid_size)
    y = torch.linspace(-5, 5, grid_size)
    z = torch.linspace(-5, 5, grid_size)
    
    # Create meshgrid - flatten in one step to reduce memory
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    Z_flat = Z.flatten()
    
    # Time snapshot
    t = torch.zeros_like(X_flat)
    
    # Generate a 3D Gaussian with some structure - compute in-place where possible
    # Optimized: Compute signal more efficiently by reusing intermediate results
    signal = torch.exp(-0.5 * (X_flat**2 + Y_flat**2 + Z_flat**2))
    signal.mul_(torch.sin(2 * X_flat))
    signal.mul_(torch.cos(2 * Y_flat))
    signal.mul_(torch.sin(2 * Z_flat))
    
    return X_flat, Y_flat, Z_flat, t, signal.unsqueeze(-1)

def generate_6d_data(grid_size=8):
    """
    Generate 6D spatiotemporal data (5D spatial + time).
    Using smaller grid due to 5D computational complexity: 8^5 = 32,768 points
    Optimized to reduce memory allocations.
    """
    # Create 5D grid
    coords = [torch.linspace(-3, 3, grid_size) for _ in range(5)]
    
    # Create meshgrid for all 5 spatial dimensions
    grids = torch.meshgrid(*coords, indexing='ij')
    
    # Flatten all dimensions - do this in a single pass
    X1_flat = grids[0].flatten()
    X2_flat = grids[1].flatten()
    X3_flat = grids[2].flatten()
    X4_flat = grids[3].flatten()
    X5_flat = grids[4].flatten()
    
    # Time snapshot
    t = torch.zeros_like(X1_flat)
    
    # Generate a 5D signal with structure - optimized to compute in-place
    # Use a combination of Gaussian envelope and oscillations in different dimensions
    r_squared = X1_flat**2 + X2_flat**2 + X3_flat**2 + X4_flat**2 + X5_flat**2
    signal = torch.exp(-0.1 * r_squared)
    signal.mul_(torch.sin(X1_flat + X2_flat))
    signal.mul_(torch.cos(X3_flat))
    signal.mul_(torch.sin(X4_flat - X5_flat))
    signal.mul_(torch.cos(X2_flat * X4_flat))
    
    return X1_flat, X2_flat, X3_flat, X4_flat, X5_flat, t, signal.unsqueeze(-1)

def train_model(model_name, model, x, t, y, epochs=2000, verbose=True, use_amp=False, grad_clip=None):
    """
    Train a model with optional optimizations.
    
    Args:
        model_name: Name of the model for logging
        model: PyTorch model to train
        x, t, y: Training data
        epochs: Number of training epochs
        verbose: Whether to print progress
        use_amp: Use automatic mixed precision training (faster on compatible GPUs)
        grad_clip: Optional gradient clipping value to prevent exploding gradients
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    losses = []
    
    # Initialize gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp and torch.cuda.is_available() else None
    
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)  # Optimized: set_to_none=True is faster
        
        # Handle different input signatures
        try:
            # Try (x, t) signature first (for Quasimoto)
            pred = model(x.squeeze(), t.squeeze()).view(-1, 1)
        except TypeError:
            # Fall back to (x) signature (for SIREN/RFF)
            pred = model(x)
            
        loss = criterion(pred, y)
        
        # Optimized backward pass with optional gradient scaling and clipping
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        losses.append(loss.item())
        
        if verbose and epoch % 500 == 0:
            print(f"[{model_name}] Epoch {epoch} Loss: {loss.item():.6f}")
    
    return loss.item(), losses

def train_model_4d(model_name, model, x, y_coord, z, t, signal, epochs=1000, verbose=True, grad_clip=None):
    """
    Training function for 4D models with optimizations.
    
    Args:
        grad_clip: Optional gradient clipping value to prevent exploding gradients
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)  # Optimized: set_to_none=True is faster
        pred = model(x, y_coord, z, t).view(-1, 1)
        loss = criterion(pred, signal)
        loss.backward()
        
        # Optimized: Optional gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        losses.append(loss.item())
        
        if verbose and epoch % 200 == 0:
            print(f"[{model_name}] Epoch {epoch} Loss: {loss.item():.6f}")
    
    return loss.item(), losses

def train_model_6d(model_name, model, x1, x2, x3, x4, x5, t, signal, epochs=500, verbose=True, grad_clip=None):
    """
    Training function for 6D models with optimizations.
    
    Args:
        grad_clip: Optional gradient clipping value to prevent exploding gradients
    """
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)  # Optimized: set_to_none=True is faster
        pred = model(x1, x2, x3, x4, x5, t).view(-1, 1)
        loss = criterion(pred, signal)
        loss.backward()
        
        # Optimized: Optional gradient clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        losses.append(loss.item())
        
        if verbose and epoch % 100 == 0:
            print(f"[{model_name}] Epoch {epoch} Loss: {loss.item():.6f}")
    
    return loss.item(), losses

def visualize_predictions(x, y_true, models, model_names, save_path='quasimoto_comparison.png'):
    """Plot true signal vs model predictions"""
    num_plots = len(models) + 1  # +1 for ground truth
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 3 * num_plots))
    
    # Handle single subplot case
    if num_plots == 1:
        axes = [axes]
    
    # Plot ground truth
    axes[0].plot(x.numpy(), y_true.numpy(), 'k-', linewidth=2, label='Ground Truth')
    axes[0].set_title('Ground Truth Signal', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    
    # Plot predictions
    for idx, (model, name) in enumerate(zip(models, model_names)):
        with torch.no_grad():
            if name.startswith("Quasimoto"):
                pred = model(x.squeeze(), torch.zeros_like(x.squeeze())).view(-1, 1)
            else:
                pred = model(x)
        
        # Calculate residual
        residual = (y_true - pred).numpy()
        mse = np.mean(residual**2)
        
        axes[idx + 1].plot(x.numpy(), y_true.numpy(), 'k--', alpha=0.3, linewidth=1, label='Ground Truth')
        axes[idx + 1].plot(x.numpy(), pred.numpy(), 'r-', linewidth=2, label=f'{name} Prediction')
        axes[idx + 1].set_title(f'{name} Fit (MSE: {mse:.8f})', fontsize=14, fontweight='bold')
        axes[idx + 1].legend(fontsize=11)
        axes[idx + 1].grid(True, alpha=0.3)
        axes[idx + 1].set_xlabel('x')
        axes[idx + 1].set_ylabel('y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    return fig

def visualize_convergence(losses_dict, save_path='quasimoto_convergence.png'):
    """Plot convergence curves for all models"""
    plt.figure(figsize=(12, 6))
    
    for name, losses in losses_dict.items():
        plt.plot(losses, label=name, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Training Convergence Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ Convergence plot saved to {save_path}")

# Execution
if __name__ == "__main__":
    print("=" * 70)
    print("QUASIMOTO EXTENDED BENCHMARK")
    print("=" * 70)
    print("\n1D Benchmark: Glitchy Chirp Signal\n")
    
    x, t, y = generate_data()
    
    # Initialize models
    print("Initializing models...")
    quasimoto_net = QuasimotoEnsemble(n=16)
    siren_net = nn.Sequential(
        SirenLayer(1, 64, is_first=True),
        SirenLayer(64, 64),
        nn.Linear(64, 1)
    )
    rff_net = RandomFourierFeatures(input_dim=1, num_features=128, sigma=5.0)
    
    print("✓ Models initialized\n")
    
    # Train models
    print("-" * 70)
    print("Training 1D Models (2000 epochs each)")
    print("-" * 70)
    
    models = [quasimoto_net, siren_net, rff_net]
    model_names = ["Quasimoto", "SIREN", "RFF"]
    losses_dict = {}
    final_losses = []
    
    for model, name in zip(models, model_names):
        print(f"\n{name}:")
        final_loss, losses = train_model(name, model, x, t, y, epochs=2000)
        losses_dict[name] = losses
        final_losses.append(final_loss)
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL RESULTS - 1D Benchmark")
    print("=" * 70)
    for name, loss in zip(model_names, final_losses):
        print(f"{name:20s} Final Loss: {loss:.8f}")
    
    # Generate visualizations
    print("\n" + "-" * 70)
    print("Generating Visualizations...")
    print("-" * 70)
    visualize_predictions(x, y, models, model_names)
    visualize_convergence(losses_dict)
    
    # 4D Benchmark
    print("\n" + "=" * 70)
    print("4D BENCHMARK: Spatiotemporal Volumetric Data")
    print("=" * 70)
    print("\nGenerating 4D data (20x20x20 grid = 8000 points)...")
    
    X, Y, Z, T, signal = generate_4d_data(grid_size=20)
    print(f"✓ 4D data generated: {len(X)} points")
    
    print("\nTraining 4D Quasimoto Ensemble (1000 epochs)...")
    quasimoto_4d = QuasimotoEnsemble4D(n=8)
    final_loss_4d, losses_4d = train_model_4d("Quasimoto-4D", quasimoto_4d, 
                                                X, Y, Z, T, signal, 
                                                epochs=1000)
    
    print("\n" + "=" * 70)
    print("FINAL RESULTS - 4D Benchmark")
    print("=" * 70)
    print(f"Quasimoto-4D Final Loss: {final_loss_4d:.8f}")
    
    # Plot 4D convergence
    plt.figure(figsize=(10, 5))
    plt.plot(losses_4d, linewidth=2, color='purple')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('MSE Loss', fontsize=12)
    plt.title('Quasimoto-4D Training Convergence', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('quasimoto_4d_convergence.png', dpi=150, bbox_inches='tight')
    print("✓ 4D convergence plot saved to quasimoto_4d_convergence.png")
    
    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • quasimoto_comparison.png - Model predictions comparison")
    print("  • quasimoto_convergence.png - Training convergence curves")
    print("  • quasimoto_4d_convergence.png - 4D model convergence")
