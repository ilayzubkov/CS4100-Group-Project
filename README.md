# Alzheimer's Disease Staging from 3D MRI

CS4100 — Artificial Intelligence | Northeastern University

This project uses convolutional neural networks to classify Alzheimer's disease severity from structural MRI brain scans. We trained and compared four different CNN architectures on the OASIS-2 longitudinal MRI dataset, with CDR scores mapped to three classes: non-demented, very mild dementia, and mild-to-moderate dementia.

---

## Repository Structure

```
CS4100-Group-Project/
├── data/
│   ├── preprocess_oasis2_first.py     # Standard preprocessing pipeline
│   ├── preprocess_oasis2_stripped.py  # Preprocessing with skull stripping
│   └── splits.json                    # Train/val/test split manifest (generated)
├── 3dcnn_baseline/
│   ├── 3d_cnn_baseline.py
│   └── results/
├── 2dcnn/
│   ├── cnn_2d.py
│   └── results/
├── 3dcnn_iterations/
│   ├── [13 intermediate model files]
│   └── results/
├── 3dcnn_best/
│   ├── 3d_cnn_best.py
│   └── results/
├── 3dcnn_with_demo/
│   ├── cs41003dcnnwithdemo.py
│   └── results/
├── explore_alzh_dataset.py
└── figures/
```

> `data/processed/` (the preprocessed `.npy` volumes, ~1.4 GB) is gitignored. You will need to run the preprocessing script before training any model.

---

## Getting Started

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU is strongly recommended — all models fall back to CPU automatically, but training is significantly slower without one

### Installation

```bash
git clone https://github.com/ilayzubkov/CS4100-Group-Project.git
cd CS4100-Group-Project
pip install torch numpy nibabel scipy openpyxl matplotlib
```

For GPU support, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/) instead.

---

## Dataset Setup

1. Download the OASIS-2 dataset from [oasis-brains.org](https://www.oasis-brains.org/). You will need:
   - `OAS2_RAW_PART1.tar.gz`
   - `OAS2_RAW_PART2.tar.gz`
   - `oasis_longitudinal_demographics.xlsx`

2. Extract both archives into the project root:

```
CS4100-Group-Project/
├── OAS2_RAW_PART1/
├── OAS2_RAW_PART2/
└── oasis_longitudinal_demographics-<hash>.xlsx
```

---

## How to Run

### Step 1 — Preprocessing

Run this once before any model. It averages repeated MRI acquisitions per session, resamples each volume to 96×96×96, applies z-score normalization, and produces a subject-level 70/15/15 train/val/test split.

```bash
python data/preprocess_oasis2_first.py
```

This writes 373 `.npy` files to `data/processed/` and a split manifest to `data/splits.json`.

For the best 3D CNN and multimodal variant, we used skull-stripped volumes. After running the standard pipeline above:

```bash
pip install antspynet
python data/preprocess_oasis2_stripped.py
```

Output is written to `data/processed_stripped/`.

---

### Step 2 — Train a Model

#### Variant 1 — Baseline 3D CNN

A 3-block volumetric CNN with manual SGD and inverse-frequency loss weighting. This is the starting point for all subsequent experiments.

```bash
python 3dcnn_baseline/3d_cnn_baseline.py
```

#### Variant 2 — 2D CNN on Axial Slices

Instead of treating MRI volumes as 3D inputs, this variant extracts 76 axial slices per volume and trains a 2D CNN on individual slices. At test time, slice-level predictions are averaged to produce one vote per session.

```bash
python 2dcnn/cnn_2d.py
```

#### Variant 3 — Regularized 3D CNN

This is the best-performing standalone 3D model. It uses a 4-block architecture with strided convolutions, batch normalization, dropout, weighted random sampling to address class imbalance, and macro accuracy checkpointing.

This model was trained on Google Colab. To run locally, update the two path variables at the top of `main()` in `3dcnn_best/3d_cnn_best.py`:

```python
splits_file = "data/splits.json"
data_dir    = "data/processed_stripped/"
```

Then run:

```bash
python 3dcnn_best/3d_cnn_best.py
```

#### Variant 4 — Multimodal CNN (3D CNN + Demographics)

Extends the 3D CNN backbone with a late-fusion demographic branch. Five patient features (age, sex, MMSE, eTIV, nWBV) are normalized and concatenated to the CNN output before the final classification layer.

This variant runs as a Colab notebook. Open it here:

> [Open in Google Colab](https://colab.research.google.com/drive/1rFIW0OhyeCby2AfkDpte0BUG2mX9eUII)

Upload `data/splits.json` and your skull-stripped volumes to Google Drive and update the path variables in the notebook before running.

#### 3D CNN Iteration History

The `3dcnn_iterations/` folder contains the 13 intermediate architectures explored between the baseline and the final best model — each change in optimizer, regularization, sampling strategy, or augmentation is its own file. These can each be run with the same setup as Variant 1 and are included for full reproducibility of the development process.

---

## Team

1. Ilay Zubkov
2. Jacob Shechter
3. Francesca Caldarella
