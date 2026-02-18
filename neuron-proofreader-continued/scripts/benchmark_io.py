"""
Benchmark I/O timing in the training pipeline.

Measures time spent in:
- Graph construction
- GT matching
- Per-proposal image I/O (S3 reads)
- Feature extraction (CPU)
- Model forward/backward
- Full epoch

Modes:
  --no-cache (default): Instrumented baseline, measures S3 and feature time separately
  --cache:              Uses in-memory feature cache from patched_reader
"""

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np
import torch

from patched_reader import (
    patch_tensorstore_reader,
    patch_swc_reader,
    patch_image_feature_extractor,
    patch_graph_loader,
)
patch_tensorstore_reader()
patch_swc_reader()
patch_graph_loader()

from neuron_proofreader.config import Config, ProposalGraphConfig, MLConfig
from neuron_proofreader.split_proofreading.split_datasets import (
    FragmentsDatasetCollection,
)
from neuron_proofreader.split_proofreading.split_feature_extraction import (
    ImageFeatureExtractor,
)
from neuron_proofreader.utils import img_util

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(REPO_ROOT)), "neuro_data", "653980")
FRAGMENTS_DIR = os.path.join(DATA_DIR, "fragments_subset")
GT_DIR = os.path.join(DATA_DIR, "gt_subset")
IMG_PATH = (
    "s3://aind-open-data/"
    "exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24/"
    "fused.zarr/0"
)


def instrument_image_feature_extractor():
    """Replace ImageFeatureExtractor.__call__ with a timed sequential version (no cache)."""
    timings = {"s3_read": [], "feature_extract": [], "crop_sizes": []}

    def timed_init_extractor(self, subgraph, proposal):
        center, shape = self.compute_crop(proposal)
        timings["crop_sizes"].append(shape)

        # Time S3 reads
        t0 = time.perf_counter()
        offset = img_util.get_offset(center, shape)
        img = self.read_image(center, shape)
        mask = self.read_segmentation(center, shape)
        t_s3 = time.perf_counter() - t0
        timings["s3_read"].append(t_s3)

        # Time feature extraction
        t0 = time.perf_counter()
        from neuron_proofreader.split_proofreading.split_feature_extraction import (
            PatchFeatureExtractor,
        )
        extractor = PatchFeatureExtractor(
            self.graph, img, mask, proposal, offset, self.patch_shape
        )
        t_feat = time.perf_counter() - t0
        timings["feature_extract"].append(t_feat)

        return extractor

    def timed_call(self, subgraph, features):
        patches, profiles = dict(), dict()
        for proposal in subgraph.proposals:
            extractor = timed_init_extractor(self, subgraph, proposal)
            profile = extractor.get_intensity_profile()
            patch = extractor.get_input_patch()
            profiles[proposal] = profile
            patches[proposal] = patch
        features.set_features(patches, "proposal_patches")
        features.integrate_proposal_profiles(profiles)

    ImageFeatureExtractor.__call__ = timed_call
    return timings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--search-radius", type=float, default=75)
    parser.add_argument("--cache", choices=["none", "memory", "disk"],
                        default="none",
                        help="Feature cache mode: none, memory, or disk")
    args = parser.parse_args()

    if args.cache == "memory":
        patch_image_feature_extractor(cache=True)
        timings = None
        print("\n*** CACHE MODE: features cached in memory after first epoch ***")
    elif args.cache == "disk":
        patch_image_feature_extractor(cache="disk")
        timings = None
        print("\n*** CACHE MODE: features cached on disk after first epoch ***")
    else:
        # Use instrumented (no cache) extractor for detailed timing
        timings = instrument_image_feature_extractor()
        print("\n*** BASELINE MODE: no caching, instrumented timing ***")

    config = Config(
        ProposalGraphConfig(anisotropy=(1.0, 1.0, 1.0), min_size=40, prune_depth=24),
        MLConfig(batch_size=32, patch_shape=(96, 96, 96), device="cpu", transform=False),
    )

    # --- Stage 1: Graph construction ---
    t0 = time.perf_counter()
    collection = FragmentsDatasetCollection()
    collection.add_dataset(
        key="653980_block",
        fragments_path=FRAGMENTS_DIR,
        img_path=IMG_PATH,
        config=config,
        gt_path=GT_DIR,
    )
    t_graph = time.perf_counter() - t0
    print(f"\n=== Graph construction: {t_graph:.2f}s ===")

    # --- Stage 2: Proposal generation + GT matching ---
    t0 = time.perf_counter()
    collection.generate_proposals(search_radius=args.search_radius)
    t_proposals = time.perf_counter() - t0
    n_proposals = collection.n_proposals()
    print(f"=== Proposal gen + GT matching: {t_proposals:.2f}s ({n_proposals} proposals) ===")

    # --- Stage 3: Epoch iteration (train-like) ---
    from neuron_proofreader.machine_learning.gnn_models import VisionHGAT
    from flexible_trainer import FlexibleTrainer

    model = VisionHGAT(patch_shape=(96, 96, 96))
    trainer = FlexibleTrainer(
        model, "benchmark", output_dir=os.path.join(REPO_ROOT, "results"),
        device="cpu", lr=1e-3, max_epochs=args.epochs,
    )

    epoch_times = []
    io_per_epoch = []
    compute_per_epoch = []
    train_crops = []

    for epoch in range(args.epochs):
        if timings:
            timings["s3_read"].clear()
            timings["feature_extract"].clear()
            timings["crop_sizes"].clear()

        t_epoch_start = time.perf_counter()

        # Train pass
        t0 = time.perf_counter()
        trainer.train_step(collection, epoch)
        t_train = time.perf_counter() - t0

        if timings:
            train_s3 = sum(timings["s3_read"])
            train_feat = sum(timings["feature_extract"])
            train_crops = list(timings["crop_sizes"])
            timings["s3_read"].clear()
            timings["feature_extract"].clear()
            timings["crop_sizes"].clear()
        else:
            train_s3 = 0
            train_feat = 0

        # Val pass
        t0 = time.perf_counter()
        trainer.validate_step(collection, epoch)
        t_val = time.perf_counter() - t0

        if timings:
            val_s3 = sum(timings["s3_read"])
            val_feat = sum(timings["feature_extract"])
        else:
            val_s3 = 0
            val_feat = 0

        t_epoch = time.perf_counter() - t_epoch_start

        total_s3 = train_s3 + val_s3
        total_feat = train_feat + val_feat

        epoch_times.append(t_epoch)
        io_per_epoch.append(total_s3)

        print(f"\nEpoch {epoch}:  total={t_epoch:.2f}s")
        print(f"  Train: {t_train:.2f}s  Val: {t_val:.2f}s")
        if timings:
            pct_io = 100 * total_s3 / t_epoch if t_epoch > 0 else 0
            print(f"  S3 reads:    {total_s3:.2f}s  ({pct_io:.1f}% of epoch)")
            print(f"  Feature ext: {total_feat:.3f}s")
            total_compute = t_epoch - total_s3 - total_feat
            compute_per_epoch.append(total_compute)
            print(f"  Compute:     {total_compute:.2f}s")

    # Cache info (if caching)
    if args.cache != "none":
        n_cached, cache_bytes = ImageFeatureExtractor._feature_cache_info()
        print(f"\n  Feature cache: {n_cached} entries, {cache_bytes / 1e6:.1f} MB")

    # Summary
    total = sum(epoch_times)
    total_io = sum(io_per_epoch)
    print(f"\n{'='*60}")
    print(f"SUMMARY ({args.epochs} epochs, {n_proposals} proposals, "
          f"cache={args.cache})")
    print(f"  Total time:    {total:.2f}s")
    print(f"  Avg epoch:     {np.mean(epoch_times):.2f}s")

    if timings:
        total_compute = sum(compute_per_epoch)
        print(f"  S3 I/O:        {total_io:.2f}s  ({100*total_io/total:.1f}%)")
        print(f"  Other compute: {total_compute:.2f}s  ({100*total_compute/total:.1f}%)")
        print(f"  Avg S3/epoch:  {np.mean(io_per_epoch):.2f}s")

    if epoch_times:
        print(f"\n  Epoch 0:       {epoch_times[0]:.2f}s (cold)")
        if len(epoch_times) > 1:
            warm_avg = np.mean(epoch_times[1:])
            print(f"  Epochs 1+:     {warm_avg:.2f}s avg (warm)")
            if epoch_times[0] > 0:
                speedup = epoch_times[0] / warm_avg
                print(f"  Speedup:       {speedup:.1f}x")

    if timings:
        if timings["crop_sizes"]:
            crops = timings["crop_sizes"]
        elif train_crops:
            crops = train_crops
        else:
            crops = []
        if crops:
            sizes = [s[0] for s in crops]
            print(f"\n  Crop sizes: min={min(sizes)}, max={max(sizes)}, "
                  f"median={int(np.median(sizes))}")
            mem_per_patch = [2 * s[0] * s[1] * s[2] * 4 / 1e6 for s in crops]
            print(f"  Patch memory: min={min(mem_per_patch):.1f}MB, "
                  f"max={max(mem_per_patch):.1f}MB per proposal")
            print(f"  Total cache (all {n_proposals}): "
                  f"{sum(mem_per_patch[:n_proposals]):.1f}MB")


if __name__ == "__main__":
    main()
