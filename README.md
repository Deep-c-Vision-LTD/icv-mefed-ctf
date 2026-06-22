# Coarse-to-Fine Compound Facial Expression Classification (iCV-MEFED)

Code and pre-computed features for the paper:

> **Improved Coarse to Fine Compound Facial Expression Classification Using Geometric and Appearance Based Features**
> S. M. Jiddah, H. Soyel, K. Yurtkan — *Signal, Image and Video Processing* (under review)

This repository lets a reviewer reproduce **every reported number and all six confusion
matrices from a single consistent run**, either directly from the provided feature files
(minutes) or end-to-end from the raw dataset.

---

## Method in one paragraph

Each face image is represented by a **1035-dimensional** feature vector: a **1000-d** deep
texture vector from **GoogLeNet** (pre-trained on ImageNet, used frozen; features taken from
the final fully-connected *loss3-classifier* layer) concatenated with a **35-d** OpenFace
**action-unit** vector. A linear SVM classifies the fused vectors. The 50 compound classes
are organised into a coarse-to-fine hierarchy: a Level-1 8-class (basic-emotion) stage, a
second-level grouping into three clusters (Angry+Contempt, Fear+Surprise,
Disgust+Happy+Sad+Neutral), and third-level classifiers over each group's fine classes.
Evaluation uses sample-mixed 80:20 5-fold cross-validation.

---

## Quick reproduction (from provided features, ~10-30 min)

```bash
pip install -r requirements.txt
python reproduce_results.py      # baseline + coarse-to-fine + all 6 confusion matrices
python multiseed.py              # 5-seed robustness (mean +/- std)
```

`reproduce_results.py` loads `features/aligned_combined_*.npy` and writes the six confusion
matrices to `figures/`. Expected console output:

```
Baseline 50-class (Fig S1):   37.82%   (P=37.91 R=37.82 F1=37.81)
Level-1 8-class   (Fig 8):    50.47%
Group 1/2/3       (Figs 10-12): 50.45% / 48.03% / 49.75%
Coarse-to-fine    (Fig 13):   49.47%   (P=49.57 R=49.47 F1=49.49)
Improvement from clustering:  +11.65 points
```

`multiseed.py` reports stability across five random partitions:

```
Baseline       : 37.28% +/- 0.34%
Coarse-to-fine : 48.93% +/- 0.28%
Improvement    : +11.64 points
```

---

## Full reproduction (from the raw dataset)

The feature files can be regenerated from scratch with `extract_features.py`. This requires
the iCV-MEFED dataset (not redistributed here), which ships the OpenFace-aligned face crops
and the OpenFace action-unit CSVs. Point the script at the extracted dataset and run:

```bash
export ICV_MEFED_DIR="/path/to/iCV MEFED/emotion/openface"
python extract_features.py       # rewrites features/aligned_combined_*.npy
python reproduce_results.py
```

---

## Repository layout

```
reproduce_results.py     main reproduction: numbers + all six confusion matrices
multiseed.py             5-seed robustness (mean +/- std)
svm_eval.py              flexible evaluator (kernel / C / split / softmax options)
extract_features.py      build features from the raw dataset (optional)
features/
    aligned_combined_X.npy        (22943 x 1035) fused features
    aligned_combined_y.npy        50-class labels
    aligned_combined_subj.npy     subject ids (for subject-independent analysis)
    aligned_combined_fnames.npy   source filenames
figures/                 output confusion matrices
requirements.txt
```

The feature file `aligned_combined_X.npy` is ~91 MB and is committed directly (within
GitHub's 100 MB per-file limit). For repositories where larger binaries are expected,
tracking the `.npy` files with **Git LFS** (`git lfs install`) is recommended.

---

## Notes on the evaluation protocol

- The reported results use **sample-mixed 80:20 5-fold cross-validation**, the standard
  protocol on this dataset.
- The third-level (coarse-to-fine) accuracy is computed by evaluating each group's
  classifier on its own fine-grained classes, as described in the manuscript.
- `aligned_combined_subj.npy` is included so that a **subject-independent** evaluation can
  also be carried out (`svm_eval.py --split subject`), should reviewers wish to assess
  cross-subject generalisation.

## License / data

The code is released for research reproducibility. The iCV-MEFED dataset is the property of
its original authors and is not redistributed here.
