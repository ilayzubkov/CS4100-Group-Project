"""
Task #2 — Open MRI Files in Python
Alzheimer's Disease Classification Project
Assigned: shechterj

Dataset: Mendeley ch87yswbz4 (2D JPG slices, 256x256, 4-class)
Note: Dataset is pre-converted from DICOM to JPG. NIfTI loading
      utilities are included at the bottom for future ADNI/OASIS-3
      integration with 3D CNN experiments.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from collections import defaultdict

# ── CONFIG ────────────────────────────────────────────────────────────────────
BASE        = Path(r"C:\Users\jacob\Downloads\Alzheimers disease dataset\Alzheimers disease dataset")
DATA_ROOT   = BASE / "OriginalDataset"
AUG_ROOT    = BASE / "AugmentedAlzheimerDataset"  # used later for training
TRAIN_DIR   = DATA_ROOT
TEST_DIR    = DATA_ROOT
CLASSES     = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
IMG_SIZE    = (256, 256)

# ── 1. VERIFY DIRECTORY STRUCTURE ─────────────────────────────────────────────
def verify_structure(root: Path) -> dict[str, int]:
    """Count images per class directly inside root (no train/test split)."""
    counts = {}
    for cls in CLASSES:
        cls_dir = root / cls
        if cls_dir.exists():
            n = len(list(cls_dir.glob("*.jpg")))
            counts[cls] = n
        else:
            counts[cls] = 0
            print(f"  [WARNING] Missing: {cls_dir}")
    return counts

print("=== Directory Structure Verification ===")
counts = verify_structure(DATA_ROOT)
total = sum(counts.values())
print(f"\nTotal images: {total}")
for cls, n in counts.items():
    bar = "█" * (n // 50)
    print(f"  {cls:<22} {n:>5}  {bar}")

# ── 2. LOAD A SINGLE IMAGE ─────────────────────────────────────────────────────
def load_image(path: Path, size: tuple = IMG_SIZE) -> np.ndarray:
    """Open a JPG MRI slice, convert to grayscale, return as float32 array [0,1]."""
    img = Image.open(path).convert("L")          # L = grayscale
    img = img.resize(size, Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0

# Quick sanity check on one image
sample_path = next((TRAIN_DIR / CLASSES[0]).glob("*.jpg"))
sample = load_image(sample_path)
print(f"\n=== Single Image Check ===")
print(f"  Path  : {sample_path.name}")
print(f"  Shape : {sample.shape}")
print(f"  dtype : {sample.dtype}")
print(f"  Range : [{sample.min():.3f}, {sample.max():.3f}]")
print(f"  Mean  : {sample.mean():.3f}  Std: {sample.std():.3f}")

# ── 3. VISUALIZE ONE SAMPLE PER CLASS ─────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Sample MRI Slices — One Per Class (Train Set)", fontsize=14, fontweight="bold")

for ax, cls in zip(axes, CLASSES):
    cls_dir = TRAIN_DIR / cls
    img_path = next(cls_dir.glob("*.jpg"))
    img = load_image(img_path)
    ax.imshow(img, cmap="gray")
    ax.set_title(cls, fontsize=10)
    ax.axis("off")

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/sample_slices.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → figures/sample_slices.png")

# ── 4. VISUALIZE CLASS DISTRIBUTION ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Class Distribution — Full Dataset", fontsize=13, fontweight="bold")

vals  = [counts[c] for c in CLASSES]
short = ["Non", "VeryMild", "Mild", "Moderate"]
bars  = ax.bar(short, vals, color=["#4CAF50","#2196F3","#FF9800","#F44336"])
ax.set_title(f"OriginalDataset (n={sum(vals)})")
ax.set_ylabel("Image Count")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
            str(v), ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("figures/class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → figures/class_distribution.png")

# ── 5. INTENSITY STATISTICS PER CLASS ─────────────────────────────────────────
print("\n=== Per-Class Pixel Intensity Statistics (Train, 20 samples each) ===")
print(f"{'Class':<24} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 60)

for cls in CLASSES:
    cls_dir = TRAIN_DIR / cls
    paths   = list(cls_dir.glob("*.jpg"))[:20]   # sample 20 to keep it fast
    imgs    = np.stack([load_image(p) for p in paths])
    print(f"{cls:<24} {imgs.mean():>8.4f} {imgs.std():>8.4f} "
          f"{imgs.min():>8.4f} {imgs.max():>8.4f}")

# ── 6. NIFTI / DICOM LOADING (for future 3D data) ─────────────────────────────
# If you later download full 3D volumes from ADNI or OASIS-3 in NIfTI format,
# use the following utilities instead of the JPG loader above.
#
# Install: pip install nibabel pydicom
#
# import nibabel as nib
# import pydicom
#
# def load_nifti(path: str) -> tuple[np.ndarray, object]:
#     """Load a .nii or .nii.gz file. Returns (voxel_array, affine_matrix)."""
#     nii   = nib.load(path)
#     vol   = nii.get_fdata(dtype=np.float32)   # shape: (X, Y, Z)
#     affine = nii.affine                        # voxel-to-world transform
#     print(f"  Volume shape : {vol.shape}")
#     print(f"  Voxel spacing: {nib.affines.voxel_sizes(affine)}")
#     return vol, affine
#
# def load_dicom_series(folder: str) -> np.ndarray:
#     """Load a folder of .dcm slices and stack into a 3D volume."""
#     slices = sorted(
#         [pydicom.dcmread(os.path.join(folder, f))
#          for f in os.listdir(folder) if f.endswith(".dcm")],
#         key=lambda s: float(s.ImagePositionPatient[2])
#     )
#     vol = np.stack([s.pixel_array for s in slices]).astype(np.float32)
#     return vol   # shape: (Z, H, W)
#
# Example usage:
#   vol, affine = load_nifti("data/ADNI/sub-001/ses-01/anat/sub-001_T1w.nii.gz")
#   plt.imshow(vol[:, :, vol.shape[2]//2], cmap="gray")  # mid axial slice

print("\nDone. All figures saved to figures/")