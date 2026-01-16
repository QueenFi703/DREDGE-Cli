import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# --- CREDITS ---
# Quasimoto Wave Function Architecture by: QueenFi703
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

# --- BENCHMARK TASK: The "Glitchy Chirp" ---
def generate_data():
    x = torch.linspace(-10, 10, 1000).view(-1, 1)
    t = torch.zeros_like(x) # Static snapshot for this test
    # A chirp: frequency increases with x. Plus a local phase glitch at x=2.
    y = torch.sin(0.5 * x**2) * torch.exp(-0.1 * x**2)
    # The Glitch - positioned at 50-55% of the signal
    glitch_start = int(0.5 * len(x))
    glitch_end = int(0.55 * len(x))
    y[glitch_start:glitch_end] += 0.5 * torch.sin(20 * x[glitch_start:glitch_end])
    return x, t, y

def train_model(model_name, model, x, t, y, epochs=2000):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Handle different input signatures - check if model accepts (x, t) or just (x)
        try:
            # Try (x, t) signature first (for Quasimoto)
            pred = model(x.squeeze(), t.squeeze()).view(-1, 1)
        except TypeError:
            # Fall back to (x) signature (for SIREN)
            pred = model(x)
            
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 500 == 0:
            print(f"[{model_name}] Epoch {epoch} Loss: {loss.item():.6f}")
    return loss.item()

# Execution
if __name__ == "__main__":
    x, t, y = generate_data()

    # 1. Quasimoto (QueenFi703) - Using a small ensemble for parity
    class QuasimotoEnsemble(nn.Module):
        def __init__(self, n=16):
            super().__init__()
            self.waves = nn.ModuleList([QuasimotoWave() for _ in range(n)])
            self.head = nn.Linear(n, 1)
        def forward(self, x, t):
            # Compute all waves efficiently
            feats = torch.stack([w(x, t) for w in self.waves], dim=-1)
            return self.head(feats)

    quasimoto_net = QuasimotoEnsemble(n=16)
    
    # 2. SIREN
    siren_net = nn.Sequential(
        SirenLayer(1, 64, is_first=True),
        SirenLayer(64, 64),
        nn.Linear(64, 1)
    )

    print("Starting Benchmarks...\n")
    q_loss = train_model("Quasimoto", quasimoto_net, x, t, y)
    s_loss = train_model("SIREN", siren_net, x, t, y)

    print(f"\nFinal Results:")
    print(f"Quasimoto (QueenFi703) Final Loss: {q_loss:.8f}")
    print(f"SIREN Final Loss: {s_loss:.8f}")
