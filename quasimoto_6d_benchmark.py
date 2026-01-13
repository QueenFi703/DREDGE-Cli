import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- CREDITS ---
# Quasimoto Wave Function Architecture by: QueenFi703
# 6D Extension: 5D Spatial + Temporal Dimensions
# ----------------

# Import classes from extended benchmark module
from quasimoto_extended_benchmark import (
    QuasimotoWave6D, QuasimotoEnsemble6D,
    generate_6d_data, train_model_6d
)

print("=" * 70)
print("QUASIMOTO 6D BENCHMARK")
print("5D Spatial + Temporal Dimensions")
print("=" * 70)

# Generate 6D data
print("\nGenerating 6D data (8^5 = 32,768 points)...")
X1, X2, X3, X4, X5, T, signal = generate_6d_data(grid_size=8)
print(f"✓ 6D data generated: {len(X1):,} points")
print(f"  Dimensions: x1, x2, x3, x4, x5, t")
print(f"  Signal shape: {signal.shape}")

# Initialize 6D Quasimoto Ensemble
print("\nInitializing Quasimoto-6D Ensemble (6 waves)...")
quasimoto_6d = QuasimotoEnsemble6D(n=6)

# Count parameters
total_params = sum(p.numel() for p in quasimoto_6d.parameters())
print(f"✓ Model initialized with {total_params:,} parameters")
print(f"  Parameters per wave: 17 (5 wave numbers, 5 velocities, 5 modulations, A, omega)")

# Train the model
print("\n" + "-" * 70)
print("Training Quasimoto-6D (500 epochs)")
print("-" * 70)

final_loss_6d, losses_6d = train_model_6d(
    "Quasimoto-6D", 
    quasimoto_6d,
    X1, X2, X3, X4, X5, T, 
    signal,
    epochs=500,
    verbose=True
)

# Results
print("\n" + "=" * 70)
print("FINAL RESULTS - 6D Benchmark")
print("=" * 70)
print(f"Quasimoto-6D Final Loss: {final_loss_6d:.8f}")
print(f"Data points processed: {len(X1):,}")
print(f"Effective dimension: 6 (5 spatial + 1 temporal)")

# Visualize convergence
print("\n" + "-" * 70)
print("Generating Visualization...")
print("-" * 70)

plt.figure(figsize=(12, 6))
plt.plot(losses_6d, linewidth=2, color='darkblue')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title('Quasimoto-6D Training Convergence (5D Spatial + Time)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('quasimoto_6d_convergence.png', dpi=150, bbox_inches='tight')
print("✓ 6D convergence plot saved to quasimoto_6d_convergence.png")

# Test prediction on a subset
print("\n" + "-" * 70)
print("Testing Prediction Quality...")
print("-" * 70)

with torch.no_grad():
    # Predict on first 100 points
    pred_subset = quasimoto_6d(X1[:100], X2[:100], X3[:100], X4[:100], X5[:100], T[:100])
    signal_subset = signal[:100]
    mse_subset = torch.mean((pred_subset.view(-1, 1) - signal_subset)**2).item()
    
print(f"Sample MSE (first 100 points): {mse_subset:.8f}")

# Generate a 2D projection visualization (x1 vs x2, averaging over other dims)
print("\n" + "-" * 70)
print("Generating 2D Projection Visualization...")
print("-" * 70)

# Create a 2D slice at x3=0, x4=0, x5=0
slice_size = 50
x1_slice = torch.linspace(-3, 3, slice_size)
x2_slice = torch.linspace(-3, 3, slice_size)
X1_grid, X2_grid = torch.meshgrid(x1_slice, x2_slice, indexing='ij')
X1_flat_slice = X1_grid.flatten()
X2_flat_slice = X2_grid.flatten()
X3_flat_slice = torch.zeros_like(X1_flat_slice)
X4_flat_slice = torch.zeros_like(X1_flat_slice)
X5_flat_slice = torch.zeros_like(X1_flat_slice)
T_flat_slice = torch.zeros_like(X1_flat_slice)

with torch.no_grad():
    pred_slice = quasimoto_6d(X1_flat_slice, X2_flat_slice, X3_flat_slice, 
                               X4_flat_slice, X5_flat_slice, T_flat_slice)
    pred_grid = pred_slice.reshape(slice_size, slice_size)

# Plot the 2D projection
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
im = ax.contourf(X1_grid.numpy(), X2_grid.numpy(), pred_grid.numpy(), 
                 levels=20, cmap='viridis')
plt.colorbar(im, ax=ax, label='Signal Value')
ax.set_xlabel('x1', fontsize=12)
ax.set_ylabel('x2', fontsize=12)
ax.set_title('Quasimoto-6D: 2D Projection (x3=x4=x5=0, t=0)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('quasimoto_6d_projection.png', dpi=150, bbox_inches='tight')
print("✓ 2D projection plot saved to quasimoto_6d_projection.png")

print("\n" + "=" * 70)
print("6D BENCHMARK COMPLETE")
print("=" * 70)
print("\nGenerated files:")
print("  • quasimoto_6d_convergence.png - Training convergence curve")
print("  • quasimoto_6d_projection.png - 2D projection (x1 vs x2)")
print("\nKey Achievements:")
print(f"  • Successfully trained on {len(X1):,} 6D points")
print(f"  • Final loss: {final_loss_6d:.8f}")
print("  • Demonstrated scalability to high-dimensional spaces")
print("\nUse Cases for 6D:")
print("  • Multi-modal sensor fusion (RGB-D + thermal + audio + time)")
print("  • High-dimensional physics simulations")
print("  • Quantum systems with multiple degrees of freedom")
print("  • Advanced climate modeling with multiple variables")
