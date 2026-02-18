"""
Inference script for the split proofreading pipeline.

Loads a trained model and runs the InferencePipeline on the filtered
subregion fragments. Works on both CPU and CUDA.

Usage:
    python scripts/run_inference.py --model-path results/session-XXXXX/model.pth
"""

import argparse
import glob
import os
import sys

# Add src/ to path for our custom modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import torch

# Apply monkey-patches before any dataset construction
from patched_reader import (
    patch_tensorstore_reader,
    patch_swc_reader,
    patch_image_feature_extractor,
    patch_graph_loader,
)
patch_tensorstore_reader()
patch_swc_reader()
patch_image_feature_extractor(cache=False)  # inference: single pass, no cache needed
patch_graph_loader()

from neuron_proofreader.config import Config, ProposalGraphConfig, MLConfig
from neuron_proofreader.machine_learning.gnn_models import VisionHGAT
from neuron_proofreader.split_proofreading.split_inference import InferencePipeline
from neuron_proofreader.utils import ml_util
from diagnostics import plot_prediction_histogram

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(REPO_ROOT)), "neuro_data", "653980")
FRAGMENTS_DIR = os.path.join(DATA_DIR, "fragments_subset")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# S3 image path
IMG_PATH = (
    "s3://aind-open-data/"
    "exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24/"
    "fused.zarr/0"
)


def patch_inference_predict_for_cpu():
    """
    Monkey-patch InferencePipeline.predict to work on CPU.

    The original uses torch.cuda.amp.autocast(enabled=True), which is
    CUDA-specific. On CPU we skip autocast entirely.
    """
    original_predict = InferencePipeline.predict

    def patched_predict(self, data):
        if self.config.ml.device == "cpu":
            with torch.inference_mode():
                x = data.get_inputs().to(self.config.ml.device)
                logits = self.model(x)

                assert logits.ndim == 2 and logits.shape[1] == 1, (
                    f"Model logits shape {logits.shape}, expected (N, 1)"
                )
                assert torch.isfinite(logits).all(), (
                    f"Model logits contain NaN/Inf"
                )

                hat_y = torch.sigmoid(logits)

                assert (hat_y >= 0).all() and (hat_y <= 1).all(), (
                    f"Sigmoid output outside [0, 1]: "
                    f"min={hat_y.min().item():.6f}, max={hat_y.max().item():.6f}"
                )

            assert hasattr(data, "idxs_proposals"), (
                "HeteroGraphData missing idxs_proposals attribute"
            )
            idx_to_id = data.idxs_proposals.idx_to_id
            hat_y = ml_util.tensor_to_list(hat_y)
            n_proposals = data.n_proposals()
            assert len(hat_y) == n_proposals, (
                f"Prediction count {len(hat_y)} != proposal count {n_proposals}"
            )
            return {idx_to_id[i]: y_i for i, y_i in enumerate(hat_y)}
        else:
            return original_predict(self, data)

    InferencePipeline.predict = patched_predict


def find_best_model(results_dir):
    """Find the best model checkpoint in the results directory."""
    pattern = os.path.join(results_dir, "session-*", "*.pth")
    pth_files = glob.glob(pattern)
    if not pth_files:
        return None
    # Sort by modification time, newest first
    pth_files.sort(key=os.path.getmtime, reverse=True)
    return pth_files[0]


def main():
    parser = argparse.ArgumentParser(description="Run split proofreading inference")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model .pth file. "
                             "If not specified, uses the most recent checkpoint.")
    parser.add_argument("--search-radius", type=float, default=100,
                        help="Search radius for proposal generation (microns)")
    parser.add_argument("--anisotropy", type=float, nargs=3,
                        default=[1.0, 1.0, 1.0],
                        help="Anisotropy (x, y, z). Use (1,1,1) since SWC coords are already in µm.")
    args = parser.parse_args()

    # Find model
    model_path = args.model_path
    if model_path is None:
        model_path = find_best_model(RESULTS_DIR)
        if model_path is None:
            print("ERROR: No model checkpoint found in results/. "
                  "Run run_training.py first or specify --model-path.")
            sys.exit(1)
    assert os.path.isfile(model_path), f"Model file not found: {model_path}"
    assert model_path.endswith(".pth"), f"Model file is not .pth: {model_path}"
    assert os.path.getsize(model_path) > 0, f"Model file is empty: {model_path}"
    print(f"Model: {model_path}")

    # Verify data
    if not os.path.isdir(FRAGMENTS_DIR):
        print(f"ERROR: Fragments not found at {FRAGMENTS_DIR}")
        sys.exit(1)
    n_frags = len([f for f in os.listdir(FRAGMENTS_DIR) if f.endswith(".swc")])
    assert n_frags > 0, f"No SWC files in {FRAGMENTS_DIR}"
    print(f"Fragments: {n_frags} SWC files")

    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Apply CPU inference patch
    if device == "cpu":
        patch_inference_predict_for_cpu()

    # Configure
    patch_shape = (96, 96, 96)
    config = Config(
        ProposalGraphConfig(anisotropy=tuple(args.anisotropy)),
        MLConfig(device=device, patch_shape=patch_shape, batch_size=8),
    )

    # Output directory
    output_dir = os.path.join(RESULTS_DIR, "inference")
    os.makedirs(output_dir, exist_ok=True)

    # Initialize model and pipeline
    print("\n--- Running Inference Pipeline ---")
    model = VisionHGAT(patch_shape=patch_shape)

    pipeline = InferencePipeline(
        fragments_path=FRAGMENTS_DIR,
        img_path=IMG_PATH,
        model_path=model_path,
        output_dir=output_dir,
        model=model,
        config=config,
    )

    # Run
    pipeline(search_radius=args.search_radius)

    # Prediction histogram
    import json

    preds_path = os.path.join(output_dir, "proposal_predictions.json")
    if os.path.isfile(preds_path):
        with open(preds_path) as f:
            preds = json.load(f)
        y_scores = list(preds.values())
        print(f"\n--- Prediction Diagnostics ---")
        plot_prediction_histogram(
            y_scores,
            save_path=os.path.join(output_dir, "prediction_histogram.png"),
        )
    else:
        print(f"\nNo proposal_predictions.json found, skipping histogram.")

    # Report results
    print(f"\n--- Results ---")
    print(f"Output directory: {output_dir}")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f} ({size / 1e3:.1f} KB)")


if __name__ == "__main__":
    main()
