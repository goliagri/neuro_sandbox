# I/O Time Improvement Experiments

## Problem
Training 30 epochs on 13 proposals takes ~20 minutes. Features are re-fetched
from S3 every epoch (60 passes: 30 train + 30 val).

## Goal
Reduce I/O % of runtime to <= 50% (already close!) AND reduce absolute runtime.

---

## Exp 0: Baseline Timing

**Command**: `python scripts/benchmark_io.py --epochs 3`

**Results**:
| Metric | Value |
|--------|-------|
| Total time (3 epochs) | 243.8s |
| S3 I/O | 100.9s (41.4%) |
| Other compute | 141.7s (58.1%) |
| Avg epoch | 81.3s |
| Avg S3/epoch | 33.7s |
| Proposals | 13 |

**Per-proposal crop sizes**: min=101, max=253, median=145
**Per-proposal patch memory**: min=8.2MB, max=129.6MB
**Total feature cache for all 13**: 449.9MB

**Breakdown per epoch** (26 S3 reads: 13 train + 13 val):
- S3 reads: ~34s (~1.3s per read)
- CPU compute: ~47s (dominated by 3D resize from crop→96³, profile extraction)
- GNN forward/backward: small fraction

**Key insight**: The "compute" time is high because crops can be up to 253³
(~16M voxels) which then get resized to 96³. The resize itself is expensive.

**Projected 30-epoch time**: 81.3 × 30 ≈ 40 min
(Our actual 30-epoch training took ~20 min — the benchmark overhead inflates this,
but the proportions should be representative.)

---

## Improvement Ideas (prioritized)

1. **In-memory feature cache** — cache (patch, profile) per proposal. ~450MB for 13 proposals. Eliminates BOTH S3 reads AND resize compute for epochs > 0. For larger runs: LRU eviction or disk spillover.

2. **Disk-based feature cache** — for larger runs where 450MB × N/13 > RAM budget. Write numpy arrays to /tmp, mmap on read. Slower than RAM but much faster than S3.

3. **Crop size cap** — limit crop to 160³ or 128³. Reduces both S3 volume and resize time. May slightly affect feature quality for very long proposals.

4. **Validate less frequently** — val every 5 epochs instead of every epoch. Halves reads if validation is skipped most of the time.

5. **Larger TensorStore cache** — proposals in same spatial region share zarr chunks. Larger cache → more chunk reuse between reads.

6. **Pre-extract all features** — before training loop, extract once, store as tensors. Pure compute training afterward.

---

## Exp 1: In-memory feature cache

**Change**: Modified `patched_reader.py:patch_image_feature_extractor()` to maintain
a `dict` cache of `{proposal: (patch, profile)}`. First access per proposal does
full S3 read + resize + profile extraction; subsequent accesses return cached arrays.

**Command**: `python scripts/benchmark_io.py --epochs 3 --cache`

**Results**:
| Metric | Value |
|--------|-------|
| Total time (3 epochs) | 164.6s |
| Epoch 0 (cold) | 70.6s |
| Epoch 1 (warm) | 35.3s |
| Epoch 2 (warm) | 58.7s |
| Warm avg | 47.0s |
| Cache size | 184.0 MB (13 entries) |
| Speedup vs baseline | 1.5x overall |

**Key observations**:
- Cache stores final 96³ patches (not raw crops), so 184MB < estimated 449MB
- Warm epoch variance is high (35-59s) — possibly GC or system load
- S3 I/O is **completely eliminated** for warm epochs
- Remaining time in warm epochs is from non-feature-extraction work:
  graph iteration, data collation, GNN forward/backward, loss computation

**Projected 30-epoch improvement**:
- Baseline: 81.3 × 30 ≈ 2439s
- Cached: 70.6 + 29 × 47.0 ≈ 1434s (1.7x improvement)
- Best case: 70.6 + 29 × 35.3 ≈ 1094s (2.2x improvement)

**Memory scaling**: 184MB / 13 proposals ≈ 14.2 MB/proposal.
For 100 proposals: ~1.4 GB. For 500: ~7.1 GB. Manageable within 23GB RAM
alongside model + graph memory (~13 GB peak).

**Verdict**: Feature cache eliminates S3 I/O completely for warm epochs.

---

## Exp 2: Warm Epoch Overhead Investigation

**Question**: Warm epochs still take 35-59s with cache. Where is the time going?

**Method**: Timed data loading, forward pass, and backward pass separately for
warm cached epochs.

**Results** (warm epochs, all 13 proposals cached):
| Component | Time/epoch | % |
|-----------|-----------|---|
| Data loading (cache hit) | 0.2-0.5s | ~1% |
| Forward pass (3D CNN) | 9-13s | ~30% |
| Backward pass (gradients) | 26-36s | ~70% |

**Conclusion**: The remaining time is **pure compute** — 3D convolution
forward/backward through VisionHGAT on CPU. This is NOT an I/O problem.
The 3D CNN processes 13 × 96³ × 2ch patches through multiple Conv3D layers.
On CPU, this is inherently slow. Would need GPU or model simplification.

**I/O is now ~1% of warm epoch time.** The cache completely solved the I/O
bottleneck. The original 41% S3 I/O is eliminated after epoch 0.

---

## Exp 3: Disk-based Feature Cache

**Change**: Added `cache="disk"` mode to `patch_image_feature_extractor()`.
Stores numpy arrays as `.npy` files in `/tmp/claude/feature_cache/` keyed
by proposal hash. Falls back to S3 on cache miss.

**Command**: `python scripts/benchmark_io.py --epochs 3 --cache disk`

**Results**:
| Metric | Value |
|--------|-------|
| Total time (3 epochs) | 143.6s |
| Epoch 0 (cold) | 63.6s |
| Epochs 1+ (warm avg) | 40.0s |
| Cache on disk | 184.0 MB (13 entries) |
| Speedup vs baseline | 1.7x overall |

**Comparison with memory cache**:
| Mode | Epoch 0 | Warm avg | 3-epoch total |
|------|---------|----------|---------------|
| None (baseline) | 81.3s | 81.3s | 243.8s |
| Memory | 70.6s | 47.0s | 164.6s |
| Disk | 63.6s | 40.0s | 143.6s |

Both cache modes perform similarly — disk reads from local SSD are nearly
as fast as memory reads for this data size. Disk cache advantage: doesn't
consume RAM, so scales to any number of proposals.

**Scaling estimate for disk cache**:
- 14.2 MB/proposal on disk
- 100 proposals: 1.4 GB disk (negligible RAM)
- 1000 proposals: 14.2 GB disk (negligible RAM)
- Disk I/O for loading: ~0.01s per proposal (vs ~1.3s S3)

---

## Summary and Conclusions

### Goal achieved: I/O reduced from 41% to ~1% of warm epoch time

The in-memory feature cache completely eliminates S3 I/O for epochs > 0.
The disk cache provides the same benefit without RAM cost.

### Time breakdown after caching (warm epoch)

| Component | Time | % | Before caching |
|-----------|------|---|----------------|
| Data loading | 0.2-0.5s | ~1% | ~34s (41%) |
| Forward pass (3D CNN) | 9-13s | ~30% | same |
| Backward pass | 26-36s | ~70% | same |

### Remaining bottleneck: CPU compute, not I/O

The 3D CNN (VisionHGAT) dominates warm epoch time. Processing 13 × 96³ × 2ch
patches through Conv3D layers on CPU takes 35-45s per epoch. This is an
inherent compute cost, not an I/O problem. Solutions would be:
- GPU (would reduce to ~1-2s/epoch)
- Smaller patch size (e.g., 64³ = 3x fewer voxels)
- Simpler model architecture

### What was NOT needed (from original idea list)

- **Crop size cap**: Not needed — caching eliminates both S3 and resize cost
- **Validate less frequently**: Minor benefit — val is same speed as train
  with caching, and only runs once per epoch
- **Larger TensorStore cache**: Irrelevant — TensorStore is bypassed by cache
- **Pre-extract all features**: This IS what the cache does, integrated into
  the training loop

### Integration into training

The cache is already integrated into `patched_reader.py` and used by all
scripts that call `patch_image_feature_extractor(cache=True)`. To use:

```python
# For small runs (< 500 proposals, ~7 GB RAM)
patch_image_feature_extractor(cache=True)

# For large runs (500+ proposals)
patch_image_feature_extractor(cache="disk")

# No caching (original behavior)
patch_image_feature_extractor(cache=False)
```

### Proposed upstream changes (not implemented)

If modifying `neuron-proofreader/` directly:

1. **`split_feature_extraction.py:ImageFeatureExtractor`**: Add a
   `_cache` dict attribute and check it in `__call__` before calling
   `init_extractor`. This would eliminate the need for the monkey-patch.

2. **`split_datasets.py:FragmentsDataset.__iter__`**: Cache the
   `HeteroGraphData` object per subgraph (not just features). Would
   save the small overhead of skeleton feature extraction and graph
   construction on each epoch.

3. **`split_datasets.py:FragmentsDatasetCollection.__iter__`**: Add
   `val_every_n_epochs` parameter to skip validation on most epochs.
   Minor benefit since val is cheap with caching.
