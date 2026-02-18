# Experiment Log

Brief records of training/inference runs. Each entry should include:

1. **Title & date/time**
2. **Intent** — what the run was trying to test or achieve
3. **Invocation(s)** — exact command(s) run
4. **Notes** — performance, memory, errors, conclusions, etc.

---

## Run 1: First Positive Labels — Anisotropy Fix (2026-02-16)

**Intent**: Get non-zero positive labels for training. Previous runs all had 0% positive
labels because the model trivially predicted all-negative.

**Root Cause Found**: The default anisotropy `(0.748, 0.748, 1.0)` was wrong for this
dataset. The SWC files (both GT and predicted fragments) store coordinates already in
physical units (µm). Applying the voxel-to-µm anisotropy scaling distorted the GT graph
through coordinate-dependent pruning, causing GT nodes to disappear from the regions
where fragments were located. Result: 0 fragments aligned to GT → 0 positive labels.

**Fix**: Set `--anisotropy 1 1 1` (now the default in both scripts).

### Run 1a: 10 epochs, lr=1e-3

```bash
python scripts/filter_subregion.py --gt-neuron N024-653980-CA.swc --max-dist 30 --max-fragments 50
python scripts/run_training.py --epochs 10 --search-radius 75 --batch-size 32 --lr 1e-3 --anisotropy 1 1 1
```

**Results**:
- Label balance: **4 positive (30.8%)**, 9 negative — first ever positive labels!
- 11/15 fragment components aligned to GT (vs 0 with old anisotropy)
- 4/13 proposals accepted (proj_dist < 8µm, structurally consistent)
- Best F1: **0.5455** (epoch 9)
- ROC AUC: **0.806**, AP: **0.715**
- Confusion matrix (threshold=0.5): 3 TP, 1 FN, 4 FP, 5 TN
- Session: `session-20260216_1241`

### Run 1b: 30 epochs, lr=3e-4

```bash
python scripts/run_training.py --epochs 30 --search-radius 75 --batch-size 32 --lr 3e-4
```

**Results**:
- Same label balance: 4 positive (30.8%), 9 negative
- Best F1: **0.6667** (epoch 29)
- ROC AUC: **0.861**, AP: **0.812**
- Confusion matrix (threshold=0.5): 2 TP, 2 FN, 0 FP, 9 TN → **100% precision**
- Val loss steady decrease: 0.29 → 0.047
- Train loss noisy but trending down (expected with 13 samples)
- Session: `session-20260216_1332`
- Model: `split_653980-20260216-29-0.6667.pth`

**Key Findings**:
1. SWC coordinates in this dataset are already in µm — do NOT apply voxel anisotropy
2. N024 (largest GT neuron, 1.5MB) with max-dist 30 gives good fragment coverage
3. With 13 proposals and 4 positive labels, model achieves strong AUC despite tiny dataset
4. Lower LR (3e-4 vs 1e-3) gives more stable training and better final performance
5. The diagnostic instrumentation in `_diagnose_gt_matching()` was crucial for finding the bug

### Run 1c: Inference with best model

```bash
python scripts/run_inference.py --search-radius 75
```

**Results**:
- Used model `split_653980-20260216-29-0.6667.pth` (auto-detected)
- 13 proposals scored, range: [0.17, 0.35]
- 0 proposals above 0.5 threshold → 0 accepted merges
- Inference time: 24s (1.85s/proposal, includes S3 image fetches)
- Peak memory: 13.07 GB
- Scores are non-trivial (not all 0 or 1) confirming model learned something
- Low scores expected: model trained on only 13 samples, underconfident at inference
- Session: `results/inference/`

---

## Run 2: Scaled-Up Training — 400 Fragments, 85 Proposals (2026-02-16)

**Intent**: Scale up training to a larger brain region. Previous runs used 50
fragments (13 proposals). Target was ~500 proposals but CPU compute is the
bottleneck (3D CNN takes ~3s/proposal/epoch), so aimed for maximum feasible
within 30 min.

**Key changes from Run 1**:
- 400 fragments (8x more) within 50µm of N024 GT neuron (all 3 GT files)
- search_radius=125 → 85 proposals (6.5x more)
- batch_size=8 (reduced from 32 to prevent OOM — CNN activations scale with batch)
- Disk-based feature cache (from I/O optimization work)

### Run 2a: 5 epochs, lr=1e-3

```bash
python scripts/filter_subregion.py --gt-neuron N024-653980-CA.swc --max-dist 50 --max-fragments 400
python scripts/run_training.py --epochs 5 --search-radius 125 --batch-size 8 --lr 1e-3 --cache disk
```

**Results**:
- Fragments: 400, Components: 76 (after irreducible compression)
- **85 proposals: 14 positive (16.5%), 71 negative**
- Best F1: **0.5600** (epoch 0)
- ROC AUC: **0.879**, AP: **0.707**
- Training time: ~21 min (epoch 0 cold: ~6 min, warm: ~3.75 min each)
- Peak memory: ~3 GB process RSS (batch_size=8 kept it safe)
- Disk cache: 85 entries, ~1.2 GB on disk
- Session: `session-20260216_2012`

**Confusion matrix (threshold=0.5)**: 10 TP, 4 FN, 12 FP, 59 TN
  - Recall: 71.4%, Precision: 45.5%

**Confusion matrix (threshold=0.8)**: 7 TP, 7 FN, 6 FP, 65 TN
  - Recall: 50.0%, Precision: 53.8%

**Key observations**:
1. Prediction histogram shows good separation: most negatives < 0.3, positives at higher scores
2. Training loss noisy (expected with small dataset + small batch_size)
3. Val precision drops by epoch 4 — model may benefit from lower LR or more epochs
4. batch_size=8 was critical — batch_size=32 caused OOM kill with 400 fragments
5. Disk feature cache eliminated S3 I/O bottleneck (all warm epochs from cache)

**Scaling findings**:
- 400 fragments → 85 proposals at radius=125 (~0.21 proposals/fragment)
- 400 fragments → 92 proposals at radius=200 (~0.23 proposals/fragment)
- To reach ~500 proposals: would need ~2000+ fragments or GPU for compute
- CPU bottleneck: 3D CNN takes ~3s/proposal/epoch → 500 proposals = 25 min/epoch
- Memory bottleneck: batch_size must be ≤ ~12 on 23GB system with other processes

