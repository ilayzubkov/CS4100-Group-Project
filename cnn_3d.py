"""
cnn_3d.py
3D CNN — Variant 1 Baseline
CS4100 Group Project — Ilay Zubkov, Jacob Shechter, Francesca Caldarella

Classifies OASIS-2 MRI volumes into three CDR stages:
  0 — Non-demented  |  1 — Very mild  |  2 — Mild-to-moderate

Sections:
  1 — Dataset   : loads preprocessed .npy volumes and labels
  2 — Model     : 3D CNN architecture
  3 — Training  : loss function, gradient descent loop, evaluation
  4 — Main      : hyperparameters, runs training, prints results
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Use GPU if available, otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# SECTION 1: DATASET

class OASISDataset(Dataset):

    def __init__(self, splits_file: str, split: str, data_dir: str = None):
        with open(splits_file) as f:
            self.records = json.load(f)[split]
        # data_dir overrides path prefix for Colab or other machines
        self.data_dir = Path(data_dir) if data_dir else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        if self.data_dir:
            path = self.data_dir / record["path"].replace("\\", "/").split("/")[-1]
        else:
            path = record["path"]
        # Load volume, add channel dim: (96,96,96) -> (1,96,96,96)
        volume = torch.tensor(np.load(path).astype(np.float32)).unsqueeze(0)
        label  = torch.tensor(record["label"], dtype=torch.long)
        return volume.to(DEVICE), label.to(DEVICE)


# SECTION 2: MODEL
#
# Three convolutional blocks, each: Conv3d -> ReLU -> MaxPool3d
# Each block doubles the number of filters while halving spatial dimensions.
# A fully connected layer then maps the extracted features to 3 class scores.
#
# Spatial progression:
#   Input        (1,  96, 96, 96)
#   After block1 (8,  48, 48, 48)
#   After block2 (16, 24, 24, 24)
#   After block3 (32, 12, 12, 12)
#   Flatten      (55296,)
#   FC hidden    (64,)
#   Output       (3,)

class CNN3D(nn.Module):

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=1,  out_channels=8,  kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=8,  out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )

        # 32 * 12 * 12 * 12 = 55296 flattened features -> 64 hidden -> 3 outputs
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 12 * 12, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x   # raw logits, shape (batch, 3)


# SECTION 3: TRAINING

def categorical_cross_entropy(logits, targets, class_weights):
    # Weighted softmax CCE: rare classes get higher weights so the model
    # is penalised more for misclassifying them.
    # weight = total_samples / (num_classes * samples_in_class)
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp    = torch.exp(logits)
    probs  = exp / exp.sum(dim=1, keepdim=True)
    correct_probs    = probs[torch.arange(len(targets)), targets]
    per_example_loss = -torch.log(correct_probs + 1e-8)
    weights = class_weights[targets]
    return (per_example_loss * weights).mean()


def accuracy(logits, targets):
    return (logits.argmax(dim=1) == targets).float().mean().item()


def per_class_accuracy(logits, targets, num_classes=3):
    predictions = logits.argmax(dim=1)
    results = {}
    for c in range(num_classes):
        mask    = targets == c
        total   = mask.sum().item()
        correct = (predictions[mask] == c).sum().item() if total > 0 else 0
        results[c] = (correct, total)
    return results


def train_one_epoch(model, loader, learning_rate, class_weights):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for volumes, labels in loader:
        logits = model(volumes)
        loss   = categorical_cross_entropy(logits, labels, class_weights)
        loss.backward()
        # Manual gradient descent: w = w - learning_rate * grad
        with torch.no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    param.data -= learning_rate * param.grad
        model.zero_grad()
        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)
    n = len(loader)
    return total_loss / n, total_acc / n


def evaluate(model, loader, class_weights):
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for volumes, labels in loader:
            logits = model(volumes)
            total_loss += categorical_cross_entropy(logits, labels, class_weights).item()
            all_logits.append(logits)
            all_labels.append(labels)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return total_loss / len(loader), accuracy(all_logits, all_labels), per_class_accuracy(all_logits, all_labels)


# SECTION 4: MAIN

def main():
    import csv
    from datetime import datetime

    # Hyperparameters
    splits_file   = "data/splits.json"
    learning_rate = 3e-4
    batch_size    = 4
    epochs        = 50

    results_dir  = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"variant1_baseline_{timestamp}.csv"

    # Set data_dir to override paths in splits.json
    # Colab: data_dir = "/content/drive/MyDrive/CS4100/processed"
    # Local: data_dir = None
    data_dir = None

    train_loader = DataLoader(OASISDataset(splits_file, "train", data_dir), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(OASISDataset(splits_file, "val",   data_dir), batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(OASISDataset(splits_file, "test",  data_dir), batch_size=batch_size, shuffle=False)

    # Class weights: total / (num_classes * count_per_class)
    train_labels  = [r["label"] for r in train_loader.dataset.records]
    class_counts  = [train_labels.count(c) for c in range(3)]
    total         = len(train_labels)
    class_weights = torch.tensor([total / (3 * c) for c in class_counts], dtype=torch.float32).to(DEVICE)

    model = CNN3D().to(DEVICE)
    print(f"Device: {DEVICE}")
    print(f"Class weights: {class_weights.tolist()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_loader.dataset)} sessions, validating on {len(val_loader.dataset)}\n")

    epoch_rows = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc       = train_one_epoch(model, train_loader, learning_rate, class_weights)
        val_loss,   val_acc, val_pc = evaluate(model, val_loader, class_weights)
        pc_str = "  ".join(f"c{c}: {val_pc[c][0]}/{val_pc[c][1]}" for c in sorted(val_pc))
        print(f"Epoch {epoch:>2}/{epochs} | train loss {train_loss:.4f}  acc {train_acc:.3f} | val loss {val_loss:.4f}  acc {val_acc:.3f} | {pc_str}")
        epoch_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_c0": f"{val_pc[0][0]}/{val_pc[0][1]}",
            "val_c1": f"{val_pc[1][0]}/{val_pc[1][1]}",
            "val_c2": f"{val_pc[2][0]}/{val_pc[2][1]}",
        })

    # Test evaluation — run once only after all training decisions are final
    print("\nFinal test evaluation:")
    test_loss, test_acc, test_pc = evaluate(model, test_loader, class_weights)
    pc_str = "  ".join(f"c{c}: {test_pc[c][0]}/{test_pc[c][1]}" for c in sorted(test_pc))
    print(f"  test loss {test_loss:.4f}  acc {test_acc:.3f} | {pc_str}")

    fieldnames = [
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_c0", "val_c1", "val_c2",
        "test_loss", "test_acc", "test_c0", "test_c1", "test_c2"
    ]
    for row in epoch_rows:
        row.update({"test_loss": "", "test_acc": "", "test_c0": "", "test_c1": "", "test_c2": ""})
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)
        writer.writerow({
            "epoch": "TEST", "train_loss": "", "train_acc": "",
            "val_loss": "", "val_acc": "", "val_c0": "", "val_c1": "", "val_c2": "",
            "test_loss": test_loss, "test_acc": test_acc,
            "test_c0": f"{test_pc[0][0]}/{test_pc[0][1]}",
            "test_c1": f"{test_pc[1][0]}/{test_pc[1][1]}",
            "test_c2": f"{test_pc[2][0]}/{test_pc[2][1]}",
        })
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
