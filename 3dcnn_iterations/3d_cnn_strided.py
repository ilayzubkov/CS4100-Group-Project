# 4 conv blocks 1->8->16->32->64 with strided Conv3d(stride=2) replacing MaxPool, BatchNorm3d + Dropout3d(0.2).
# Strided conv lets gradients flow through all spatial positions; MaxPool only passes through the max.

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OASISDataset(Dataset):

    def __init__(self, splits_file: str, split: str,
                 data_dir: str = None, augment: bool = False):
        with open(splits_file) as f:
            self.records = json.load(f)[split]
        self.data_dir = Path(data_dir) if data_dir else None
        self.augment  = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        if self.data_dir:
            path = self.data_dir / Path(record["path"]).name
        else:
            path = record["path"]

        volume = torch.tensor(np.load(path).astype(np.float32)).unsqueeze(0)

        if self.augment and torch.rand(1).item() < 0.5:
            volume = torch.flip(volume, dims=[3])

        label = torch.tensor(record["label"], dtype=torch.long)
        return volume.to(DEVICE), label.to(DEVICE)


# Spatial progression (stride=2 per block, same dims as MaxPool version):
#   Input        (1,  96, 96, 96)
#   After block1 (8,  48, 48, 48)
#   After block2 (16, 24, 24, 24)
#   After block3 (32, 12, 12, 12)
#   After block4 (64,  6,  6,  6)
#   Flatten      13824 -> FC(64) -> 3

class CNN3D(nn.Module):

    def __init__(self, dropout_p: float = 0.2):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=1,  out_channels=8,  kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=8,  out_channels=16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 6 * 6 * 6, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.classifier(x)
        return x   # raw logits, shape (batch, 3)


def categorical_cross_entropy(logits, targets, class_weights):
    """
    Manual softmax cross-entropy with per-class loss scaling.

    Logits are shifted by their row-wise max before exponentiation for numerical
    stability. Each example's loss is scaled by its class weight.
    """
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp    = torch.exp(logits)
    probs  = exp / exp.sum(dim=1, keepdim=True)

    correct_probs    = probs[torch.arange(len(targets)), targets]
    per_example_loss = -torch.log(correct_probs + 1e-8)

    weights = class_weights[targets]
    return (per_example_loss * weights).mean()


def accuracy(logits, targets):
    """Overall fraction correct."""
    return (logits.argmax(dim=1) == targets).float().mean().item()


def per_class_accuracy(logits, targets, num_classes=3):
    """Returns {class: (correct, total)} for each class."""
    predictions = logits.argmax(dim=1)
    results = {}
    for c in range(num_classes):
        mask    = targets == c
        total   = mask.sum().item()
        correct = (predictions[mask] == c).sum().item() if total > 0 else 0
        results[c] = (correct, total)
    return results


def train_one_epoch(model, loader, optimizer, class_weights):
    """One pass over the training set. Returns (mean_loss, mean_acc)."""
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for volumes, labels in loader:
        optimizer.zero_grad()
        logits = model(volumes)
        loss   = categorical_cross_entropy(logits, labels, class_weights)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


def evaluate(model, loader, class_weights):
    """Evaluate on val/test split without updating weights."""
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
    acc        = accuracy(all_logits, all_labels)
    per_class  = per_class_accuracy(all_logits, all_labels)

    return total_loss / len(loader), acc, per_class


def main():
    import csv
    from datetime import datetime

    splits_file   = "data/splits.json"
    learning_rate = 1e-3
    batch_size    = 4
    epochs        = 50
    dropout_p     = 0.2

    results_dir  = Path("/content/drive/MyDrive/results")   # local: Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"strided_{timestamp}.csv"

    data_dir = None   # Colab: "/content/drive/MyDrive/CS4100/processed"

    train_loader = DataLoader(
        OASISDataset(splits_file, "train", data_dir, augment=True),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        OASISDataset(splits_file, "val",  data_dir, augment=False),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        OASISDataset(splits_file, "test", data_dir, augment=False),
        batch_size=batch_size, shuffle=False
    )

    train_labels  = [r["label"] for r in train_loader.dataset.records]
    class_counts  = [train_labels.count(c) for c in range(3)]
    total         = len(train_labels)
    class_weights = torch.tensor(
        [total / (3 * c) for c in class_counts], dtype=torch.float32
    ).to(DEVICE)
    print(f"Class weights: {class_weights.tolist()}")

    model     = CNN3D(dropout_p=dropout_p).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_loader.dataset)} sessions (augmented), "
          f"validating on {len(val_loader.dataset)}\n")

    epoch_rows = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc       = train_one_epoch(model, train_loader, optimizer, class_weights)
        val_loss,   val_acc, val_pc = evaluate(model, val_loader, class_weights)

        pc_str = "  ".join(f"c{c}: {val_pc[c][0]}/{val_pc[c][1]}" for c in sorted(val_pc))
        print(f"Epoch {epoch:>2}/{epochs} | "
              f"train loss {train_loss:.4f}  acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f}  acc {val_acc:.3f} | {pc_str}")

        epoch_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "val_c0": f"{val_pc[0][0]}/{val_pc[0][1]}",
            "val_c1": f"{val_pc[1][0]}/{val_pc[1][1]}",
            "val_c2": f"{val_pc[2][0]}/{val_pc[2][1]}",
        })

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
