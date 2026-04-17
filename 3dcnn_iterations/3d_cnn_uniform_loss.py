# 4 conv blocks 1->4->8->16->32 with strided Conv3d(stride=2), BatchNorm3d + Dropout3d(0.4), ~241K params.
# Equal loss weights (ones): WRS already balances class frequency; adding sqrt weights on top over-pushes rare classes.

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

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

        # When the same C2 volume is sampled multiple times per epoch by the
        # WeightedRandomSampler, the flip ensures each draw is a different variant.
        if self.augment and torch.rand(1).item() < 0.5:
            volume = torch.flip(volume, dims=[3])

        label = torch.tensor(record["label"], dtype=torch.long)
        return volume.to(DEVICE), label.to(DEVICE)


# Spatial progression:
#   Input        (1,  96, 96, 96)
#   After block1 (4,  48, 48, 48)   <- stride=2
#   After block2 (8,  24, 24, 24)   <- stride=2
#   After block3 (16, 12, 12, 12)   <- stride=2
#   After block4 (32,  6,  6,  6)   <- stride=2
#   Flatten      6912 -> FC(32) -> 3

class CNN3D(nn.Module):

    def __init__(self, dropout_p: float = 0.4):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv3d(in_channels=1,  out_channels=4,  kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(4),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )
        self.block2 = nn.Sequential(
            nn.Conv3d(in_channels=4,  out_channels=8,  kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )
        self.block3 = nn.Sequential(
            nn.Conv3d(in_channels=8,  out_channels=16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )
        self.block4 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Dropout3d(p=dropout_p)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 6 * 6 * 6, 32),
            nn.ReLU(),
            nn.Linear(32, 3)
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
    Manual softmax cross-entropy with per-class scaling.

    With class_weights=ones(3) this reduces to standard unweighted cross-entropy.
    Logits are shifted by their row-wise max before exponentiation for numerical stability.
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


def macro_accuracy(per_class_dict):
    """
    Mean of per-class accuracies.

    Gives each class equal weight regardless of size. Used as the checkpoint criterion
    because val has only 6 C2 examples (11%) while test has 13 (25%) — overall accuracy
    underweights C2 as a signal for which checkpoint generalises best.
    """
    accs = [c / t for c, t in per_class_dict.values() if t > 0]
    return sum(accs) / len(accs)


def train_one_epoch(model, loader, optimizer, class_weights, max_grad_norm):
    """One pass over training with gradient clipping. Returns (mean_loss, mean_acc)."""
    model.train()
    total_loss, total_acc = 0.0, 0.0

    for volumes, labels in loader:
        optimizer.zero_grad()
        logits = model(volumes)
        loss   = categorical_cross_entropy(logits, labels, class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_acc  += accuracy(logits, labels)

    n = len(loader)
    return total_loss / n, total_acc / n


def evaluate(model, loader, class_weights):
    """
    Evaluate on val/test split without updating weights.

    Returns (mean_loss, overall_acc, macro_acc, per_class_dict).
    """
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
    overall    = accuracy(all_logits, all_labels)
    per_class  = per_class_accuracy(all_logits, all_labels)
    macro      = macro_accuracy(per_class)

    return total_loss / len(loader), overall, macro, per_class


def main():
    import csv
    from datetime import datetime

    splits_file = "/content/drive/MyDrive/CS4100/splits.json"
    data_dir    = "/content/processed_stripped2"

    learning_rate = 1e-4
    weight_decay  = 1e-3
    batch_size    = 4
    epochs        = 100
    dropout_p     = 0.4
    max_grad_norm = 1.0

    results_dir  = Path("/content/drive/MyDrive/CS4100/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"uniform_loss_{timestamp}.csv"

    # Val and test loaders are never resampled
    val_loader = DataLoader(
        OASISDataset(splits_file, "val",  data_dir, augment=False),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        OASISDataset(splits_file, "test", data_dir, augment=False),
        batch_size=batch_size, shuffle=False
    )

    train_dataset = OASISDataset(splits_file, "train", data_dir, augment=True)
    train_labels  = [r["label"] for r in train_dataset.records]
    class_counts  = [train_labels.count(c) for c in range(3)]
    total         = len(train_labels)

    # WeightedRandomSampler: weight each sample by 1/class_count so all classes
    # are drawn roughly equally. C2 goes from ~25 draws/epoch to ~total/3 ≈ 89.
    sample_weights = torch.tensor(
        [1.0 / class_counts[r["label"]] for r in train_dataset.records],
        dtype=torch.float32
    )
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=total, replacement=True)
    # shuffle must be False when using a custom sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    # Equal loss weights: the sampler already enforces balanced class exposure.
    # Adding sqrt inverse-frequency weights on top (~6.7x total C2-vs-C0 push)
    # caused the model to ignore C0 for the first 17 epochs in the best variant.
    class_weights = torch.ones(3, dtype=torch.float32).to(DEVICE)
    print(f"Class weights (loss): {class_weights.tolist()}")
    print(f"Class counts: C0={class_counts[0]}  C1={class_counts[1]}  C2={class_counts[2]}")
    print(f"Expected draws/epoch (oversampled): ~{total // 3} per class")

    model     = CNN3D(dropout_p=dropout_p).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    print(f"Device: {DEVICE}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"lr={learning_rate}  weight_decay={weight_decay}  grad_clip={max_grad_norm}  dropout={dropout_p}")
    print(f"Training on {len(train_dataset)} sessions (oversampled + augmented), "
          f"validating on {len(val_loader.dataset)}\n")

    epoch_rows         = []
    best_macro_val_acc = 0.0
    best_epoch         = -1
    best_model_state   = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, class_weights, max_grad_norm
        )
        val_loss, val_overall, val_macro, val_pc = evaluate(model, val_loader, class_weights)

        checkpoint_marker = ""
        if val_macro > best_macro_val_acc:
            best_macro_val_acc = val_macro
            best_epoch         = epoch
            best_model_state   = {k: v.clone() for k, v in model.state_dict().items()}
            checkpoint_marker  = " *"

        pc_str = "  ".join(f"c{c}: {val_pc[c][0]}/{val_pc[c][1]}" for c in sorted(val_pc))
        print(f"Epoch {epoch:>3}/{epochs} | "
              f"train loss {train_loss:.4f}  acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f}  acc {val_overall:.3f}  macro {val_macro:.3f} | "
              f"{pc_str}{checkpoint_marker}")

        epoch_rows.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_overall, "val_macro": val_macro,
            "val_c0": f"{val_pc[0][0]}/{val_pc[0][1]}",
            "val_c1": f"{val_pc[1][0]}/{val_pc[1][1]}",
            "val_c2": f"{val_pc[2][0]}/{val_pc[2][1]}",
            "checkpoint": "best" if val_macro == best_macro_val_acc else "",
        })

    print(f"\nBest macro val acc {best_macro_val_acc:.4f} at epoch {best_epoch} — restoring for test.")
    model.load_state_dict(best_model_state)

    print("\nFinal test evaluation (best macro val checkpoint):")
    test_loss, test_overall, test_macro, test_pc = evaluate(model, test_loader, class_weights)
    pc_str = "  ".join(f"c{c}: {test_pc[c][0]}/{test_pc[c][1]}" for c in sorted(test_pc))
    print(f"  test loss {test_loss:.4f}  acc {test_overall:.3f}  macro {test_macro:.3f} | {pc_str}")

    fieldnames = [
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_macro", "val_c0", "val_c1", "val_c2", "checkpoint",
        "test_loss", "test_acc", "test_macro", "test_c0", "test_c1", "test_c2"
    ]
    for row in epoch_rows:
        row.update({
            "test_loss": "", "test_acc": "", "test_macro": "",
            "test_c0": "", "test_c1": "", "test_c2": ""
        })

    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)
        writer.writerow({
            "epoch": "TEST", "train_loss": "", "train_acc": "",
            "val_loss": "", "val_acc": "", "val_macro": "",
            "val_c0": "", "val_c1": "", "val_c2": "", "checkpoint": "",
            "test_loss": test_loss, "test_acc": test_overall, "test_macro": test_macro,
            "test_c0": f"{test_pc[0][0]}/{test_pc[0][1]}",
            "test_c1": f"{test_pc[1][0]}/{test_pc[1][1]}",
            "test_c2": f"{test_pc[2][0]}/{test_pc[2][1]}",
        })
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
