"""
cnn_2d.py
2D CNN on Axial Slices — Variant 2
CS4100 Group Project — Ilay Zubkov, Jacob Shechter, Francesca Caldarella

Approach:
    Each preprocessed (96, 96, 96) volume yields 76 axial slices (indices
    10–86, skipping the mostly-empty outer slices).  All slices from one
    session inherit that session's CDR label.  A 2D CNN is trained on
    individual slices; at test time, slice-level softmax scores are averaged
    per volume for a single volume-level prediction.

Sections:
  1 — Dataset      : extracts 2D axial slices from .npy volumes
  2 — Model        : 2D CNN architecture
  3 — Training     : weighted CCE (from scratch), gradient descent, evaluation
  4 — Verification : visual sanity check on slice extraction
  5 — Main         : hyperparameters, training loop, results saved to CSV

Depends on: preprocess_oasis2_first.py having been run so that
    data/processed/  contains .npy volumes
    data/splits.json contains the subject-level train/val/test manifest

Reference: LeCun et al., 1998. Gradient-Based Learning Applied to Document
           Recognition.
"""

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SLICE_START = 10   # skip outermost slices (mostly empty / skull edge)
SLICE_END   = 86   # gives 76 slices per volume


# SECTION 1: DATASET

class OASISSliceDataset(Dataset):
    """
    Yields individual 2D axial slices from preprocessed OASIS-2 volumes.

    Each item:
        slice  — float32 tensor shape (1, 96, 96)   (channel-first)
        label  — long tensor, class in {0, 1, 2}

    The split is subject-level (enforced by splits.json), so all slices from
    one subject land in the same split — no leakage.
    """

    def __init__(self, splits_file: str, split: str, data_dir: str = None):
        with open(splits_file) as f:
            records = json.load(f)[split]
        self.data_dir = Path(data_dir) if data_dir else None
        # Build flat list of (path, slice_index, label) for every slice
        self.samples = []
        for r in records:
            for i in range(SLICE_START, SLICE_END):
                self.samples.append((r["path"], i, r["label"]))
        # Keep records for volume-level evaluation
        self.records = records

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, i, label = self.samples[idx]
        if self.data_dir:
            filename = path.replace("\\", "/").split("/")[-1]
            path = self.data_dir / filename
        vol  = np.load(path)              # (96, 96, 96)  float32
        slc  = vol[:, :, i][np.newaxis]   # (1, 96, 96)
        return (
            torch.tensor(slc,  dtype=torch.float32).to(DEVICE),
            torch.tensor(label, dtype=torch.long).to(DEVICE),
        )


# SECTION 2: MODEL
#
# Three convolutional blocks, each: Conv2d -> ReLU -> MaxPool2d
# Spatial progression:
#   Input        (1,  96, 96)
#   After block1 (16, 48, 48)
#   After block2 (32, 24, 24)
#   After block3 (64, 12, 12)
#   Flatten      (9216,)
#   FC hidden    (128,)
#   Output       (3,)

class CNN2D(nn.Module):

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # 64 * 12 * 12 = 9216 flattened features -> 128 hidden -> 3 outputs
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
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
    # Matches cnn_3d.py exactly; numerically stabilised by subtracting max.
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
    for slices, labels in loader:
        logits = model(slices)
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
    """Slice-level evaluation — mirrors evaluate() in cnn_3d.py."""
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []
    with torch.no_grad():
        for slices, labels in loader:
            logits = model(slices)
            total_loss += categorical_cross_entropy(logits, labels, class_weights).item()
            all_logits.append(logits)
            all_labels.append(labels)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return total_loss / len(loader), accuracy(all_logits, all_labels), per_class_accuracy(all_logits, all_labels)


def volume_level_accuracy(model, records, data_dir=None):
    """
    Aggregate slice-level softmax scores per volume, then argmax.
    This is the primary comparable metric against the 3D CNN — each subject
    gets one vote regardless of how many slices it contributes.
    """
    model.eval()
    correct = 0
    data_dir = Path(data_dir) if data_dir else None
    for r in records:
        filename = r["path"].replace("\\", "/").split("/")[-1]
        path = (data_dir / filename) if data_dir else r["path"]
        vol    = np.load(path)
        slices = vol[:, :, SLICE_START:SLICE_END]          # (96, 96, 76)
        slices = slices.transpose(2, 0, 1)[:, np.newaxis]  # (76, 1, 96, 96)
        tensor = torch.tensor(slices, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            probs = model(tensor).softmax(dim=1).mean(dim=0)  # average over slices
        pred = probs.argmax().item()
        if pred == r["label"]:
            correct += 1
    return correct / len(records)


# SECTION 4: VERIFICATION

def verify_slicing(splits_file, data_dir=None):
    """
    Visual sanity check that slice extraction is working correctly.

    Row 1 — 10 evenly spaced slices across the FULL depth (0-95).
             Red border = skipped, green border = kept.
    Row 2 — 10 evenly spaced slices from the KEPT range (10-85).
    Row 3 — One slice (z=48) from a sample volume in each split,
             to confirm train/val/test loaded correctly.

    Saves to figures/verify_slicing.png and displays inline.
    """
    with open(splits_file) as f:
        splits = json.load(f)

    data_dir = Path(data_dir) if data_dir else None

    def load_vol(record):
        filename = record["path"].replace("\\", "/").split("/")[-1]
        path = (data_dir / filename) if data_dir else record["path"]
        return np.load(path)

    train_record = splits["train"][0]
    vol    = load_vol(train_record)
    label  = train_record["label"]
    mri_id = train_record["mri_id"]

    full_indices = np.linspace(0, 95, 10, dtype=int)
    kept_indices = np.linspace(SLICE_START, SLICE_END - 1, 10, dtype=int)

    split_slices = {}
    for name, recs in splits.items():
        if recs:
            v = load_vol(recs[0])
            split_slices[name] = (v[:, :, 48], recs[0]["label"], recs[0]["mri_id"])

    fig, axes = plt.subplots(3, 10, figsize=(20, 7))
    fig.suptitle(f"Slice extraction verification — {mri_id}  label={label}",
                 fontsize=13, fontweight="bold")

    # Row 1: full depth
    for ax, idx in zip(axes[0], full_indices):
        color = "red" if (idx < SLICE_START or idx >= SLICE_END) else "green"
        ax.imshow(vol[:, :, idx], cmap="gray", vmin=-3, vmax=3)
        ax.set_title(f"z={idx}", fontsize=8, color=color)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2); spine.set_visible(True)
    axes[0, 0].set_ylabel("Full depth\n(red=skipped)", fontsize=9)

    # Row 2: kept range
    for ax, idx in zip(axes[1], kept_indices):
        ax.imshow(vol[:, :, idx], cmap="gray", vmin=-3, vmax=3)
        ax.set_title(f"z={idx}", fontsize=8, color="green")
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor("green"); spine.set_linewidth(2); spine.set_visible(True)
    axes[1, 0].set_ylabel(f"Kept (z={SLICE_START}–{SLICE_END-1})", fontsize=9)

    # Row 3: one slice per split
    for col, (name, (slc, lbl, mid)) in enumerate(split_slices.items()):
        ax = axes[2, col]
        ax.imshow(slc, cmap="gray", vmin=-3, vmax=3)
        ax.set_title(f"{name}\nlabel={lbl}", fontsize=8)
        ax.axis("off")
    for col in range(len(split_slices), 10):
        axes[2, col].set_visible(False)
    axes[2, 0].set_ylabel("One vol per split\n(z=48)", fontsize=9)

    red_patch   = mpatches.Patch(color="red",   label=f"Skipped (z<{SLICE_START} or z≥{SLICE_END})")
    green_patch = mpatches.Patch(color="green", label=f"Kept (z={SLICE_START}–{SLICE_END-1})")
    fig.legend(handles=[red_patch, green_patch], loc="lower right", fontsize=9)

    plt.tight_layout()
    Path("figures").mkdir(exist_ok=True)
    out = Path("figures/verify_slicing.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → {out}")


# SECTION 5: MAIN

def main():
    # Hyperparameters
    splits_file   = "data/splits.json"
    verify_slices = True   # set to False to skip the visual check
    learning_rate = 3e-4
    batch_size    = 64
    epochs        = 30

    # Set data_dir to override paths in splits.json
    # Colab: data_dir = "/content/drive/MyDrive/CS4100/processed"
    data_dir = "data/processed"

    if verify_slices:
        verify_slicing(splits_file, data_dir)

    results_dir  = Path("results")
    results_dir.mkdir(exist_ok=True)
    timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"variant2_2dcnn_{timestamp}.csv"

    train_dataset = OASISSliceDataset(splits_file, "train", data_dir)
    val_dataset   = OASISSliceDataset(splits_file, "val",   data_dir)
    test_dataset  = OASISSliceDataset(splits_file, "test",  data_dir)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # Class weights computed from slice-level label counts in training set.
    # Every slice in a volume shares the volume's label, so the slice counts
    # are volume counts * slices_per_volume — the ratio cancels cleanly.
    # Formula: total / (num_classes * count_per_class), matching cnn_3d.py.
    train_labels  = [r["label"] for r in train_dataset.records]
    class_counts  = [train_labels.count(c) for c in range(3)]
    total         = len(train_labels)
    class_weights = torch.tensor(
        [total / (3 * c) for c in class_counts], dtype=torch.float32
    ).to(DEVICE)

    model = CNN2D().to(DEVICE)
    print(f"Device: {DEVICE}")
    print(f"Class weights: {class_weights.tolist()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {len(train_dataset)} slices "
          f"({len(train_dataset.records)} volumes), "
          f"validating on {len(val_dataset)} slices "
          f"({len(val_dataset.records)} volumes)\n")

    epoch_rows = []
    for epoch in range(1, epochs + 1):
        train_loss, train_acc          = train_one_epoch(model, train_loader, learning_rate, class_weights)
        val_loss,   val_acc,   val_pc  = evaluate(model, val_loader, class_weights)
        val_vol_acc                    = volume_level_accuracy(model, val_dataset.records, data_dir)
        pc_str = "  ".join(f"c{c}: {val_pc[c][0]}/{val_pc[c][1]}" for c in sorted(val_pc))
        print(f"Epoch {epoch:>2}/{epochs} | "
              f"train loss {train_loss:.4f}  acc {train_acc:.3f} | "
              f"val loss {val_loss:.4f}  acc {val_acc:.3f}  vol_acc {val_vol_acc:.3f} | "
              f"{pc_str}")
        epoch_rows.append({
            "epoch":        epoch,
            "train_loss":   train_loss,
            "train_acc":    train_acc,
            "val_loss":     val_loss,
            "val_acc":      val_acc,
            "val_vol_acc":  val_vol_acc,
            "val_c0":       f"{val_pc[0][0]}/{val_pc[0][1]}",
            "val_c1":       f"{val_pc[1][0]}/{val_pc[1][1]}",
            "val_c2":       f"{val_pc[2][0]}/{val_pc[2][1]}",
        })

    # Test evaluation — run once only after all training decisions are final
    print("\nFinal test evaluation:")
    test_loss, test_acc, test_pc = evaluate(model, test_loader, class_weights)
    test_vol_acc                 = volume_level_accuracy(model, test_dataset.records, data_dir)
    pc_str = "  ".join(f"c{c}: {test_pc[c][0]}/{test_pc[c][1]}" for c in sorted(test_pc))
    print(f"  test loss {test_loss:.4f}  acc {test_acc:.3f}  vol_acc {test_vol_acc:.3f} | {pc_str}")

    fieldnames = [
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_vol_acc", "val_c0", "val_c1", "val_c2",
        "test_loss", "test_acc", "test_vol_acc", "test_c0", "test_c1", "test_c2",
    ]
    for row in epoch_rows:
        row.update({"test_loss": "", "test_acc": "", "test_vol_acc": "",
                    "test_c0": "", "test_c1": "", "test_c2": ""})
    with open(results_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(epoch_rows)
        writer.writerow({
            "epoch": "TEST", "train_loss": "", "train_acc": "",
            "val_loss": "", "val_acc": "", "val_vol_acc": "",
            "val_c0": "", "val_c1": "", "val_c2": "",
            "test_loss":    test_loss,
            "test_acc":     test_acc,
            "test_vol_acc": test_vol_acc,
            "test_c0": f"{test_pc[0][0]}/{test_pc[0][1]}",
            "test_c1": f"{test_pc[1][0]}/{test_pc[1][1]}",
            "test_c2": f"{test_pc[2][0]}/{test_pc[2][1]}",
        })
    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    main()
