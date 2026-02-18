# neuron-proofreader-continued

A lightweight adaptation of the Allen Institute's [neuron-proofreader](https://github.com/AllenInstitute/neuron_morphology_tools) split-proofreading pipeline, designed to run on CPU-only workstations with bounded RAM (~23 GB).

## What it does

The upstream library automatically reconnects over-fragmented neuron skeleton reconstructions from exaSPIM whole-brain light-sheet microscopy. It builds a graph of skeleton fragments, generates merge proposals between nearby endpoints, classifies them with a heterogeneous graph attention network (VisionHGAT), and merges accepted connections.

This project wraps that pipeline with:

- **Memory-safe monkey-patches** (`src/patched_reader.py`) — replaces all unbounded `ThreadPoolExecutor` / `ProcessPoolExecutor` usage and the 2 GB TensorStore cache with sequential processing and a 100 MB cache, preventing OOM on memory-constrained systems.
- **CPU/GPU-agnostic training** (`src/flexible_trainer.py`) — subclasses the upstream `Trainer` to auto-detect the device and conditionally disable CUDA-specific mixed precision / GradScaler. Collects per-epoch metrics and generates training curve plots.
- **Data pipeline scripts** (`scripts/`) — download public exaSPIM data from S3, spatially filter fragments by proximity to ground-truth neurites (KD-tree), and run training or inference end-to-end.

## Data

Uses brain 653980 from AIND's public S3 bucket (`s3://aind-open-data/exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24/`). Imagery is read on-demand via TensorStore (never downloaded locally). SWC fragments and ground-truth skeletons are stored in `../neuro_data/653980/`.

## Usage

```bash
# 1. Download SWC fragments + ground truth from S3
python scripts/download_data.py

# 2. Filter to a trainable subregion (~50-500 fragments near GT)
python scripts/filter_subregion.py --max-dist 50 --max-fragments 2000

# 3. Train VisionHGAT model
python scripts/run_training.py --epochs 5 --search-radius 100

# 4. Run inference with trained model
python scripts/run_inference.py
```

## Key design decisions

- **Monkey-patching over forking**: Patches are applied at import time rather than maintaining a full fork, keeping the codebase minimal and easy to update with upstream changes.
- **Sequential-by-default**: Every concurrent operation is serialized. This trades throughput for predictable memory usage.
- **Proximity-based filtering**: Fragments are selected by KD-tree distance to ground-truth nodes rather than bounding boxes, ensuring training data actually overlaps with annotation targets.
- **On-demand imagery**: TensorStore reads 96^3 voxel patches directly from S3, avoiding multi-terabyte local downloads.
