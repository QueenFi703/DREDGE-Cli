import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- CREDITS ---
# Quasimoto Wave Function Architecture by: QueenFi703
# Interference Basis: Complex-valued wave interference with learned superposition
# ----------------

from quasimoto_extended_benchmark import (
    QuasimotoInterferenceBasis,
    generate_data,
    QuasimotoEnsemble,
    SirenLayer
)

print("=" * 70)
print("QUASIMOTO INTERFERENCE BASIS BENCHMARK")
print("Complex-Valued Wave Interference with Learned Superposition")
print("=" * 70)

# Generate 1D chirp data
print("\nGenerating 1D glitchy chirp signal...")
x, t, y = generate_data()
print(f"✓ Data generated: {len(x)} points")

# Initialize models
print("\nInitializing models...")

# 1. Quasimoto Interference Basis (new complex-valued architecture)
interference_model = QuasimotoInterferenceBasis(
    dim=1,              # 1D input
    num_fields=8,       # 8 coupled wave fields
    out_features=1      # 1D output
)

# 2. Original Quasimoto Ensemble for comparison
quasimoto_model = QuasimotoEnsemble(n=8)

# 3. SIREN baseline
siren_model = nn.Sequential(
    SirenLayer(1, 32, is_first=True),
    SirenLayer(32, 32),
    nn.Linear(32, 1)
)

# Count parameters
interference_params = sum(p.numel() for p in interference_model.parameters())
quasimoto_params = sum(p.numel() for p in quasimoto_model.parameters())
siren_params = sum(p.numel() for p in siren_model.parameters())

print(f"✓ Quasimoto Interference Basis: {interference_params:,} parameters")
print(f"  - {interference_model.num_fields} complex wave fields")
print(f"  - Anisotropic envelope (σ per dimension)")
print(f"  - Learned superposition weights")
print(f"✓ Quasimoto Ensemble: {quasimoto_params:,} parameters")
print(f"✓ SIREN: {siren_params:,} parameters")

# Training function
def train_model(model_name, model, x, t, y, epochs=1000):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Handle different input signatures
        if model_name == "Interference":
            # Interference basis expects x as [..., dim]
            pred = model(x, t.squeeze())
        elif model_name == "Quasimoto":
            pred = model(x.squeeze(), t.squeeze()).view(-1, 1)
        else:  # SIREN
            pred = model(x)
            
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 200 == 0:
            print(f"[{model_name}] Epoch {epoch} Loss: {loss.item():.6f}")
    
    return loss.item(), losses

# Train models
print("\n" + "-" * 70)
print("Training Models (1000 epochs each)")
print("-" * 70)

models = [
    ("Interference", interference_model),
    ("Quasimoto", quasimoto_model),
    ("SIREN", siren_model)
]

losses_dict = {}
final_losses = {}

for name, model in models:
    print(f"\n{name}:")
    final_loss, losses = train_model(name, model, x, t, y, epochs=1000)
    losses_dict[name] = losses
    final_losses[name] = final_loss

# Print results
print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
for name, loss in final_losses.items():
    print(f"{name:20s} Final Loss: {loss:.8f}")

# Generate visualizations
print("\n" + "-" * 70)
print("Generating Visualizations...")
print("-" * 70)

# 1. Predictions comparison
fig, axes = plt.subplots(len(models) + 1, 1, figsize=(14, 3 * (len(models) + 1)))

# Ground truth
axes[0].plot(x.numpy(), y.numpy(), 'k-', linewidth=2, label='Ground Truth')
axes[0].set_title('Ground Truth Signal', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')

# Model predictions
for idx, (name, model) in enumerate(models):
    with torch.no_grad():
        if name == "Interference":
            pred = model(x, t.squeeze())
        elif name == "Quasimoto":
            pred = model(x.squeeze(), t.squeeze()).view(-1, 1)
        else:
            pred = model(x)
    
    mse = torch.mean((y - pred)**2).item()
    
    axes[idx + 1].plot(x.numpy(), y.numpy(), 'k--', alpha=0.3, linewidth=1, label='Ground Truth')
    axes[idx + 1].plot(x.numpy(), pred.numpy(), 'r-', linewidth=2, label=f'{name} Prediction')
    axes[idx + 1].set_title(f'{name} Fit (MSE: {mse:.8f})', fontsize=14, fontweight='bold')
    axes[idx + 1].legend(fontsize=11)
    axes[idx + 1].grid(True, alpha=0.3)
    axes[idx + 1].set_xlabel('x')
    axes[idx + 1].set_ylabel('y')

plt.tight_layout()
plt.savefig('quasimoto_interference_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: quasimoto_interference_comparison.png")

# 2. Convergence curves
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
plt.savefig('quasimoto_interference_convergence.png', dpi=150, bbox_inches='tight')
print("✓ Saved: quasimoto_interference_convergence.png")

# 3. Analyze complex wave interference patterns
print("\n" + "-" * 70)
print("Analyzing Complex Wave Interference...")
print("-" * 70)

with torch.no_grad():
    # Get intermediate complex wave fields
    x_input = x
    t_input = t.squeeze()
    
    # Access internal parameters
    sigma = torch.exp(interference_model.log_sigma).clamp(min=1e-3)
    print(f"Anisotropic envelope width σ: {sigma.item():.4f}")
    print(f"Envelope velocity v: {interference_model.v.item():.4f}")
    print(f"\nWave field amplitudes A:")
    for i, a in enumerate(interference_model.A):
        print(f"  Field {i+1}: {a.item():.4f}")
    
    print(f"\nSuperposition weights (first 5):")
    weights = interference_model.superposition.weight.data[0]
    for i in range(min(5, len(weights))):
        print(f"  Weight {i+1}: {weights[i].item():.4f}")

print("\n" + "=" * 70)
print("BENCHMARK COMPLETE")
print("=" * 70)
print("\nKey Features of Interference Basis:")
print("  • Complex-valued wave functions with exp(i·phase)")
print("  • Coupled fields with interference patterns")
print("  • Anisotropic Gaussian envelopes (different σ per dimension)")
print("  • Learned superposition of real and imaginary parts")
print("  • Quasiperiodic phase distortion")
print("\nGenerated files:")
print("  • quasimoto_interference_comparison.png")
print("  • quasimoto_interference_convergence.png")
