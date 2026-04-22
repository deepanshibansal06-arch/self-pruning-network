"""
Self-Pruning Neural Network on CIFAR-10
========================================
Author  : [Your Name]
Date    : 2025
Task    : Tredence AI Engineering Internship — Case Study

Core idea
---------
Each weight in every Linear layer is paired with a learnable "gate" scalar.
A sigmoid squashes the gate into (0, 1), and we multiply it element-wise with
the weight before the forward pass.  An L1 penalty on all gate values is added
to the cross-entropy loss so that the optimiser has a reason to drive gates
toward exactly zero — effectively pruning weights on-the-fly during training.

The beauty of L1 (vs L2): the gradient of |x| is ±1, which provides a
constant "push" toward zero regardless of the gate's current magnitude.
L2's gradient shrinks as values approach zero, so it's bad at reaching
exactly zero. L1 doesn't care — it pushes just as hard at 0.001 as at 10.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import os

# ──────────────────────────────────────────────
# 0. Reproducibility
# ──────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] running on: {DEVICE}")


# ──────────────────────────────────────────────
# 1. PrunableLinear — the heart of the method
# ──────────────────────────────────────────────
class PrunableLinear(nn.Module):
    """
    A drop-in replacement for nn.Linear that adds learnable gate scalars.

    For each weight w_ij there is a corresponding gate_score g_ij.
    During the forward pass:
        gate     = sigmoid(g_ij)          ← squash to (0, 1)
        pruned_w = w_ij * gate            ← soft masking
        output   = pruned_w @ input + b   ← standard affine transform

    Because both `weight` and `gate_scores` are nn.Parameters, autograd
    will compute gradients for both and the optimiser will update both.

    When gate_scores → -∞, sigmoid(gate_scores) → 0, so the weight is
    effectively removed without being literally deleted from the graph.
    This keeps the computation graph intact and gradients clean.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight + bias (same init as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Gate scores — initialised near 0 so sigmoid ≈ 0.5 at the start.
        # This means no gates are initially pruned, so the network can learn
        # freely before sparsity pressure gradually closes unwanted ones.
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute soft gates in (0, 1)
        gates = torch.sigmoid(self.gate_scores)          # shape: (out, in)

        # Element-wise multiplication — gates broadcast correctly
        pruned_weights = self.weight * gates             # shape: (out, in)

        # Standard linear operation using the pruned weight matrix
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Return gate values (detached) for analysis / sparsity reporting."""
        return torch.sigmoid(self.gate_scores).detach()

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, "
                f"bias={self.bias is not None}")


# ──────────────────────────────────────────────
# 2. The Network
# ──────────────────────────────────────────────
class SelfPruningNet(nn.Module):
    """
    A simple feed-forward network for CIFAR-10 (32×32 RGB → 10 classes).

    We deliberately keep it medium-sized so there is enough redundancy for
    the pruning mechanism to have a real effect.  Too small and there's
    nothing to prune; too large and training takes forever on a laptop.
    """

    def __init__(self):
        super().__init__()
        # CIFAR-10 images: 3 × 32 × 32 = 3072 input features
        self.net = nn.Sequential(
            PrunableLinear(3072, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            PrunableLinear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            PrunableLinear(256, 10),   # output logits for 10 classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)    # flatten spatial dims
        return self.net(x)

    def prunable_layers(self):
        """Yields all PrunableLinear instances in the network."""
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                yield module

    def sparsity_loss(self) -> torch.Tensor:
        """
        L1 norm of all gate values = sum of |sigmoid(g)| over every weight.

        Since sigmoid is always positive, |gate| == gate, so this is just
        the sum of all gate values.  Minimising this pushes gates toward 0.
        """
        total = torch.tensor(0.0, device=DEVICE)
        for layer in self.prunable_layers():
            gates = torch.sigmoid(layer.gate_scores)
            total = total + gates.sum()
        return total

    def compute_sparsity(self, threshold: float = 1e-2) -> float:
        """
        Returns fraction of weights whose gate value < threshold.
        A gate below threshold is considered "pruned".
        """
        total_weights  = 0
        pruned_weights = 0
        for layer in self.prunable_layers():
            gates = layer.get_gates()
            total_weights  += gates.numel()
            pruned_weights += (gates < threshold).sum().item()
        return pruned_weights / total_weights if total_weights > 0 else 0.0

    def all_gate_values(self) -> np.ndarray:
        """Collect every gate value into a flat numpy array for plotting."""
        vals = []
        for layer in self.prunable_layers():
            vals.append(layer.get_gates().cpu().numpy().ravel())
        return np.concatenate(vals)


# ──────────────────────────────────────────────
# 3. Data Loading
# ──────────────────────────────────────────────
def get_cifar10_loaders(batch_size: int = 256):
    """Standard CIFAR-10 with mild augmentation for training."""
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,  download=True, transform=train_transform)
    test_set  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


# ──────────────────────────────────────────────
# 4. Training Loop
# ──────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, lam: float):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()

        logits = model(images)

        # ── total loss ──────────────────────────────────────────────────
        # Cross-entropy handles "how well are we classifying?"
        # Sparsity loss handles "how many gates can we turn off?"
        # Lambda balances the two objectives.
        ce_loss  = F.cross_entropy(logits, labels)
        sp_loss  = model.sparsity_loss()
        loss     = ce_loss + lam * sp_loss
        # ────────────────────────────────────────────────────────────────

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct   / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    correct = 0
    total   = 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ──────────────────────────────────────────────
# 5. Full Experiment for One λ Value
# ──────────────────────────────────────────────
def run_experiment(lam: float, epochs: int = 30) -> dict:
    print(f"\n{'='*55}")
    print(f"  Starting experiment  |  λ = {lam}")
    print(f"{'='*55}")

    train_loader, test_loader = get_cifar10_loaders()

    model     = SelfPruningNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "train_acc": [], "sparsity": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, lam)
        scheduler.step()

        sparsity = model.compute_sparsity()
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["sparsity"].append(sparsity)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:02d}/{epochs}  "
              f"loss={train_loss:.4f}  "
              f"train_acc={train_acc:.3f}  "
              f"sparsity={sparsity:.1%}  "
              f"({elapsed:.1f}s)")

    test_acc = evaluate(model, test_loader)
    final_sparsity = model.compute_sparsity()
    gate_vals = model.all_gate_values()

    print(f"\n  ► Final test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"  ► Final sparsity      : {final_sparsity:.4f}  ({final_sparsity*100:.2f}%)")

    return {
        "lam"      : lam,
        "model"    : model,
        "test_acc" : test_acc,
        "sparsity" : final_sparsity,
        "gate_vals": gate_vals,
        "history"  : history,
    }


# ──────────────────────────────────────────────
# 6. Plotting
# ──────────────────────────────────────────────
def plot_gate_distributions(results: list, save_path: str = "gate_distributions.png"):
    """
    One subplot per λ value showing the distribution of all gate values.
    A successful run shows a large spike near 0 (pruned weights) and a
    smaller cluster away from 0 (surviving weights).
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    # Find the best model (highest test accuracy) for the title marker
    best_idx = max(range(n), key=lambda i: results[i]["test_acc"])

    for ax, res in zip(axes, results):
        gate_vals = res["gate_vals"]
        lam       = res["lam"]
        sparsity  = res["sparsity"]
        test_acc  = res["test_acc"]

        # Use more bins near 0 to capture the spike detail
        bins = np.concatenate([
            np.linspace(0,    0.05, 50),
            np.linspace(0.05, 1.0,  50),
        ])

        counts, edges = np.histogram(gate_vals, bins=bins)
        # Colour: red for near-zero (pruned), blue for surviving
        colors = ["#e74c3c" if e < 0.05 else "#2980b9"
                  for e in (edges[:-1] + edges[1:]) / 2]

        ax.bar(edges[:-1], counts, width=np.diff(edges),
               color=colors, align="edge", edgecolor="none", alpha=0.85)

        ax.set_title(
            f"λ = {lam}\n"
            f"Test Acc: {test_acc*100:.1f}%  |  Sparsity: {sparsity*100:.1f}%",
            fontsize=12, fontweight="bold"
        )
        ax.set_xlabel("Gate Value (sigmoid output)", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Gate Value Distributions After Training\n"
                 "Red ≈ pruned  |  Blue = surviving connections",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[plot] saved → {save_path}")


def plot_training_curves(results: list, save_path: str = "training_curves.png"):
    """Plot sparsity growth and train accuracy over epochs for each λ."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colours = ["#e74c3c", "#f39c12", "#27ae60"]

    for res, colour in zip(results, colours):
        epochs = range(1, len(res["history"]["train_acc"]) + 1)
        lbl = f"λ={res['lam']}"
        ax1.plot(epochs, [s * 100 for s in res["history"]["sparsity"]],
                 color=colour, linewidth=2, label=lbl)
        ax2.plot(epochs, [a * 100 for a in res["history"]["train_acc"]],
                 color=colour, linewidth=2, label=lbl)

    for ax, title, ylabel in [
        (ax1, "Sparsity Growth During Training", "Sparsity (%)"),
        (ax2, "Train Accuracy During Training",  "Accuracy (%)"),
    ]:
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved → {save_path}")


# ──────────────────────────────────────────────
# 7. Main — three λ values as required
# ──────────────────────────────────────────────
if __name__ == "__main__":
    LAMBDAS = [1e-5, 1e-4, 5e-4]   # low, medium, high
    EPOCHS  = 30                    # increase to 50+ for better accuracy

    all_results = []
    for lam in LAMBDAS:
        result = run_experiment(lam, epochs=EPOCHS)
        all_results.append(result)

    # ── Summary table ──────────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SUMMARY TABLE")
    print("="*55)
    print(f"  {'Lambda':<12} {'Test Acc (%)':>14} {'Sparsity (%)':>14}")
    print(f"  {'-'*40}")
    for r in all_results:
        print(f"  {r['lam']:<12} "
              f"{r['test_acc']*100:>13.2f}% "
              f"{r['sparsity']*100:>13.2f}%")
    print("="*55)

    # ── Plots ──────────────────────────────────────────────────────────
    plot_gate_distributions(all_results, "gate_distributions.png")
    plot_training_curves(all_results,    "training_curves.png")

    print("\n[done] all experiments complete.")
