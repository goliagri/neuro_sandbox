"""
Training script for the split proofreading pipeline.

Builds a FragmentsDatasetCollection from the filtered subregion, generates
proposals, and trains a VisionHGAT model. Works on both CPU and CUDA.

Usage:
    python scripts/run_training.py [--epochs 5] [--search-radius 100]
"""

import argparse
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
# patch_image_feature_extractor is called in main() after CLI parsing
patch_graph_loader()

from neuron_proofreader.config import Config, ProposalGraphConfig, MLConfig
from neuron_proofreader.split_proofreading.split_datasets import (
    FragmentsDatasetCollection,
)
from neuron_proofreader.machine_learning.gnn_models import VisionHGAT
from flexible_trainer import FlexibleTrainer
from diagnostics import (
    report_label_balance,
    plot_prediction_histogram,
    plot_roc_pr_curves,
    print_confusion_matrix,
)


def _diagnose_gt_matching():
    """Monkey-patch groundtruth_generation.run() to log which checks fail."""
    from collections import defaultdict
    from copy import deepcopy
    import numpy as np
    import networkx as nx
    from neuron_proofreader.split_proofreading import groundtruth_generation as gt_gen
    from neuron_proofreader.utils import geometry_util, util

    _original_run = gt_gen.run

    def verbose_run(gt_graph, pred_graph):
        gt_graph.set_kdtree()
        gt_graph.set_kdtree(node_type="branching")
        pred_to_gt = gt_gen.get_pred_to_gt_mapping(gt_graph, pred_graph)

        # --- Diagnostic A: fragment alignment ---
        n_components = sum(1 for _ in nx.connected_components(pred_graph))
        n_aligned = sum(1 for v in pred_to_gt.values() if v is not None)
        print(f"\n--- GT Matching Diagnostics ---")
        print(f"  Fragment components: {n_components}")
        print(f"  Aligned to GT: {n_aligned}  (need >= 2 for any positive proposal)")

        # Show per-fragment alignment details
        for nodes in nx.connected_components(pred_graph):
            node = util.sample_once(nodes)
            comp_id = pred_graph.node_component_id[node]
            gt_id = pred_to_gt[comp_id]

            # Recompute alignment stats for logging
            dists_by_gt = defaultdict(list)
            n_pts = 0
            for edge in pred_graph.subgraph(nodes).edges:
                for xyz in pred_graph.edges[edge]["xyz"]:
                    hat_xyz = geometry_util.kdtree_query(gt_graph.kdtree, xyz)
                    gid = gt_graph.xyz_to_component_id(hat_xyz)
                    d = geometry_util.dist(hat_xyz, xyz)
                    dists_by_gt[gid].append(d)
                    n_pts += 1
            if n_pts == 0:
                print(f"    comp {comp_id}: 0 edge points (singleton?), not aligned")
                continue
            best_gid = util.find_best(dists_by_gt)
            d_arr = np.array(dists_by_gt[best_gid])
            pct_aligned = len(d_arr) / n_pts
            score_60 = np.percentile(d_arr, 60)
            status = "ALIGNED" if gt_id is not None else "NOT aligned"
            reason = ""
            if gt_id is None:
                reasons = []
                if score_60 >= 7:
                    reasons.append(f"60th-pct dist={score_60:.1f} >= 7")
                if pct_aligned <= 0.6:
                    reasons.append(f"pct_aligned={pct_aligned:.2f} <= 0.6")
                if best_gid is None:
                    reasons.append("best_gid is None")
                reason = " (" + ", ".join(reasons) + ")" if reasons else ""
            print(f"    comp {comp_id}: {status}{reason}  "
                  f"[60th-pct={score_60:.1f}µm, %aligned={pct_aligned:.2f}, "
                  f"n_pts={n_pts}, best_gt={best_gid}]")

        # --- Diagnostic B: proposal-level checks ---
        proposals = pred_graph.get_sorted_proposals()
        print(f"\n  Total proposals: {len(proposals)}")
        accepts_graph = deepcopy(pred_graph)
        gt_accepts = []
        fail_alignment = 0
        fail_proj_dist = 0
        fail_structure = 0
        accepted = 0

        for proposal in proposals:
            i, j = tuple(proposal)
            id1 = pred_graph.node_component_id[i]
            id2 = pred_graph.node_component_id[j]

            if pred_to_gt[id1] != pred_to_gt[id2] or pred_to_gt[id1] is None:
                fail_alignment += 1
                continue

            dist = gt_gen.compute_proposal_proj_dist(gt_graph, pred_graph, proposal)
            if dist > 8:
                fail_proj_dist += 1
                print(f"    FAIL proj_dist: proposal ({id1},{id2}) "
                      f"proj_dist_90th={dist:.1f}µm > 8µm")
                continue

            gt_id = pred_to_gt[id1]
            is_consistent = gt_gen.is_structure_consistent(
                gt_graph, pred_graph, accepts_graph, gt_id, proposal
            )
            if is_consistent:
                accepts_graph.add_edge(i, j)
                gt_accepts.append(proposal)
                accepted += 1
                print(f"    ACCEPT: proposal ({id1},{id2}) "
                      f"proj_dist={dist:.1f}µm")
            else:
                fail_structure += 1
                print(f"    FAIL structure: proposal ({id1},{id2}) "
                      f"proj_dist={dist:.1f}µm but structurally inconsistent")

        print(f"\n  Proposal results:")
        print(f"    Failed alignment check: {fail_alignment}")
        print(f"    Failed proj_dist > 8µm: {fail_proj_dist}")
        print(f"    Failed structure check: {fail_structure}")
        print(f"    Accepted: {accepted}")

        return gt_accepts

    gt_gen.run = verbose_run
    print("GT matching diagnostics enabled.")


# Uncomment to enable verbose GT matching diagnostics:
# _diagnose_gt_matching()

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(REPO_ROOT)), "neuro_data", "653980")
FRAGMENTS_DIR = os.path.join(DATA_DIR, "fragments_subset")
GT_DIR = os.path.join(DATA_DIR, "gt_subset")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

# S3 image path (accessed on-demand via TensorStore, not downloaded)
IMG_PATH = (
    "s3://aind-open-data/"
    "exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24/"
    "fused.zarr/0"
)


def verify_data():
    """Check that required data files exist."""
    if not os.path.isdir(FRAGMENTS_DIR):
        print(f"ERROR: Fragments not found at {FRAGMENTS_DIR}")
        print("Run filter_subregion.py first.")
        sys.exit(1)

    n_frags = len([f for f in os.listdir(FRAGMENTS_DIR) if f.endswith(".swc")])
    if n_frags == 0:
        print(f"ERROR: No SWC files in {FRAGMENTS_DIR}")
        sys.exit(1)
    print(f"Found {n_frags} fragment SWC files")

    if not os.path.isdir(GT_DIR):
        print(f"ERROR: GT not found at {GT_DIR}")
        sys.exit(1)

    n_gt = len([f for f in os.listdir(GT_DIR) if f.endswith(".swc")])
    print(f"Found {n_gt} GT SWC files")
    return n_frags


def main():
    parser = argparse.ArgumentParser(description="Train split proofreading model")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs")
    parser.add_argument("--search-radius", type=float, default=100,
                        help="Search radius for proposal generation (microns)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (max proposals per subgraph)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--anisotropy", type=float, nargs=3,
                        default=[1.0, 1.0, 1.0],
                        help="Anisotropy (x, y, z). Use (1,1,1) since SWC coords are already in µm.")
    parser.add_argument("--cache", choices=["memory", "disk", "none"],
                        default="memory",
                        help="Feature cache mode: memory (fast, ~14MB/proposal RAM), "
                             "disk (for large runs), none (re-read from S3 every epoch)")
    args = parser.parse_args()

    # Apply feature extractor patch with chosen cache mode
    cache_arg = True if args.cache == "memory" else args.cache if args.cache == "disk" else False
    patch_image_feature_extractor(cache=cache_arg)

    # Verify data
    verify_data()

    # Validate args
    assert args.epochs > 0, f"--epochs must be positive, got {args.epochs}"
    assert args.search_radius > 0, f"--search-radius must be positive, got {args.search_radius}"
    assert args.batch_size > 0, f"--batch-size must be positive, got {args.batch_size}"
    assert args.lr > 0, f"--lr must be positive, got {args.lr}"
    assert all(a > 0 for a in args.anisotropy), (
        f"--anisotropy values must be positive, got {args.anisotropy}"
    )

    # Auto-detect device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Configure
    patch_shape = (96, 96, 96)
    graph_config = ProposalGraphConfig(
        anisotropy=tuple(args.anisotropy),
        min_size=40,
        prune_depth=24,
    )
    ml_config = MLConfig(
        batch_size=args.batch_size,
        patch_shape=patch_shape,
        device=device,
        transform=False,
    )
    config = Config(graph_config, ml_config)

    # Build dataset collection
    print("\n--- Building Dataset ---")
    collection = FragmentsDatasetCollection()
    collection.add_dataset(
        key="653980_block",
        fragments_path=FRAGMENTS_DIR,
        img_path=IMG_PATH,
        config=config,
        gt_path=GT_DIR,
    )

    # Generate proposals
    print("\n--- Generating Proposals ---")
    collection.generate_proposals(search_radius=args.search_radius)
    n_proposals = collection.n_proposals()
    assert isinstance(n_proposals, int) and n_proposals >= 0, (
        f"n_proposals() returned {n_proposals!r}, expected non-negative int"
    )
    p_accepts = collection.p_accepts()
    assert isinstance(p_accepts, float) and 0.0 <= p_accepts <= 1.0, (
        f"p_accepts() returned {p_accepts!r}, expected float in [0, 1]"
    )
    print(f"Total proposals: {n_proposals}")
    print(f"Accept rate: {p_accepts:.4f}")

    print("\n--- Label Balance ---")
    report_label_balance(collection)

    if n_proposals == 0:
        print("ERROR: No proposals generated. Try adjusting --search-radius "
              "or re-running filter_subregion.py with different --padding.")
        sys.exit(1)

    # Initialize model
    print("\n--- Training ---")
    model = VisionHGAT(patch_shape=patch_shape)
    try:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,}")
    except ValueError:
        print("Model uses lazy initialization — parameter count available after first forward pass.")

    # Train
    os.makedirs(RESULTS_DIR, exist_ok=True)
    trainer = FlexibleTrainer(
        model,
        "split_653980",
        output_dir=RESULTS_DIR,
        device=device,
        lr=args.lr,
        max_epochs=args.epochs,
    )

    # Use same collection for both train and val in this quick test
    trainer.run(collection, collection)

    # Always save a checkpoint (the default Trainer only saves when F1 improves,
    # which won't happen with 0% accept rate since all labels are negative)
    trainer.save_model(args.epochs - 1)

    assert 0.0 <= trainer.best_f1 <= 1.0, (
        f"best_f1 is {trainer.best_f1}, expected value in [0, 1]"
    )
    print(f"\nTraining complete. Results saved to {trainer.log_dir}")
    print(f"Best F1: {trainer.best_f1:.4f}")

    # --- Diagnostics ---
    import math

    if trainer.last_val_logits:
        y_true = trainer.last_val_targets
        y_scores = [1 / (1 + math.exp(-l)) for l in trainer.last_val_logits]

        print("\n--- Prediction Diagnostics ---")
        plot_prediction_histogram(
            y_scores, y_true,
            save_path=os.path.join(trainer.log_dir, "prediction_histogram.png"),
        )
        plot_roc_pr_curves(
            y_true, y_scores,
            save_path=os.path.join(trainer.log_dir, "roc_pr_curves.png"),
        )
        print_confusion_matrix(y_true, y_scores, threshold=0.8)
    else:
        print("\nNo validation predictions to diagnose.")


if __name__ == "__main__":
    main()
