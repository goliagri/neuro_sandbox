"""
Spatial filtering of SWC fragments to select a small subregion.

Reads GT neuron SWC files (with OFFSET headers) and builds a KD-tree of
their node positions. Then scans the predicted SWC zip and keeps fragments
whose root coordinate is within a configurable distance of any GT node.

This proximity-based approach (vs. bounding-box) ensures selected fragments
actually overlap with GT branches, which is critical for producing nonzero
accept rates during training.

Target: ~50-500 fragments yielding ~10-100 proposals for feasible CPU training.
"""

import argparse
import os
import re
import shutil
import sys
import time
import numpy as np
from scipy.spatial import cKDTree
from zipfile import ZipFile

# Paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(REPO_ROOT)), "neuro_data", "653980")
GT_DIR = os.path.join(DATA_DIR, "gt_swcs")
SWC_ZIP = os.path.join(DATA_DIR, "swcs-processed.zip")
FRAGMENTS_DIR = os.path.join(DATA_DIR, "fragments_subset")
GT_SUBSET_DIR = os.path.join(DATA_DIR, "gt_subset")


def parse_swc_offset(filepath):
    """Parse the # OFFSET header from a Janelia Workstation SWC file."""
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            m = re.match(r"#\s*OFFSET\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)", line)
            if m:
                offset = np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                assert offset.shape == (3,), f"Offset shape {offset.shape} != (3,)"
                assert np.isfinite(offset).all(), f"Offset contains non-finite values: {offset}"
                return offset
            if line and not line.startswith("#"):
                break
    return np.zeros(3)


def parse_swc_file(filepath, offset=None):
    """Parse an SWC file and return xyz coordinates as (N, 3) array.

    If offset is provided, adds it to all coordinates to convert from
    relative to absolute coordinates.
    """
    coords = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 7:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                coords.append([x, y, z])
    arr = np.array(coords) if coords else np.zeros((0, 3))
    assert arr.ndim == 2 and arr.shape[1] == 3, (
        f"SWC coordinate array has shape {arr.shape}, expected (N, 3)"
    )
    if offset is not None and len(arr) > 0:
        arr = arr + offset
    assert np.isfinite(arr).all(), (
        f"SWC coordinates contain non-finite values in {filepath}"
    )
    return arr


def get_first_data_line_xyz(content_bytes):
    """Extract root (x, y, z) from an SWC file's first data line."""
    for line in content_bytes.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 7:
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                return None
            return x, y, z
    return None


def find_gt_neuron(gt_dir):
    """Find a GT neuron with a reasonable-sized SWC file."""
    swc_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(".swc")])
    if not swc_files:
        print(f"ERROR: No SWC files found in {gt_dir}", file=sys.stderr)
        sys.exit(1)

    # Prefer -CA.swc (complete annotation) files
    candidates = [f for f in swc_files if "-CA.swc" in f]
    if not candidates:
        candidates = [f for f in swc_files if "axon" in f.lower()]
    if not candidates:
        candidates = swc_files

    # Pick the smallest CA neuron to keep subregion manageable
    candidates.sort(key=lambda f: os.path.getsize(os.path.join(gt_dir, f)))
    chosen = candidates[0]
    print(f"Selected GT neuron: {chosen} ({os.path.getsize(os.path.join(gt_dir, chosen)) / 1e3:.0f} KB)")
    return os.path.join(gt_dir, chosen)


def load_all_gt_coords(gt_dir, neuron_prefix):
    """Load and merge coordinates from all GT SWC files for a neuron."""
    all_coords = []
    gt_files = sorted([f for f in os.listdir(gt_dir)
                       if f.endswith(".swc") and f.startswith(neuron_prefix)])
    for fn in gt_files:
        filepath = os.path.join(gt_dir, fn)
        offset = parse_swc_offset(filepath)
        coords = parse_swc_file(filepath, offset=offset)
        if len(coords) > 0:
            all_coords.append(coords)
            print(f"  {fn}: {len(coords):,} nodes, "
                  f"offset=({offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f})")
    if all_coords:
        return np.vstack(all_coords)
    return np.zeros((0, 3))


def filter_fragments_by_proximity(zip_path, gt_kdtree, max_dist, max_fragments=2000):
    """Scan zip and find SWC files whose root is within max_dist of any GT node."""
    matching = []
    distances = []
    total = 0
    t0 = time.time()

    with ZipFile(zip_path, "r") as zf:
        swc_names = [n for n in zf.namelist() if n.endswith(".swc")]
        print(f"Total SWC files in zip: {len(swc_names):,}")

        for name in swc_names:
            total += 1
            if total % 100000 == 0:
                elapsed = time.time() - t0
                print(f"  Scanned {total:,} files in {elapsed:.1f}s, "
                      f"{len(matching)} matches so far...")

            content = zf.read(name)
            xyz = get_first_data_line_xyz(content)
            if xyz is None:
                continue

            dist, idx = gt_kdtree.query(xyz)
            assert np.isfinite(dist) and dist >= 0, (
                f"KD-tree returned invalid distance {dist} for fragment {name}"
            )
            if dist <= max_dist:
                matching.append(name)
                distances.append(dist)

            if len(matching) >= max_fragments:
                print(f"  Reached max_fragments={max_fragments}, stopping scan.")
                break

    elapsed = time.time() - t0
    print(f"Scanned {total:,} files in {elapsed:.1f}s, found {len(matching)} matches.")
    if distances:
        distances = np.array(distances)
        print(f"  Distance to nearest GT node — "
              f"median: {np.median(distances):.1f}, "
              f"mean: {np.mean(distances):.1f}, "
              f"max: {np.max(distances):.1f} µm")
    return matching


def extract_fragments(zip_path, names, output_dir):
    """Extract matching SWC files from zip to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    with ZipFile(zip_path, "r") as zf:
        for name in names:
            content = zf.read(name)
            basename = os.path.basename(name)
            out_path = os.path.join(output_dir, basename)
            with open(out_path, "wb") as f:
                f.write(content)


def copy_gt_for_subset(gt_dir, gt_subset_dir, gt_neuron_path):
    """Copy the selected GT neuron's files to the subset GT directory."""
    os.makedirs(gt_subset_dir, exist_ok=True)
    neuron_prefix = os.path.basename(gt_neuron_path).split("-")[0]
    gt_files = [f for f in os.listdir(gt_dir)
                if f.endswith(".swc") and f.startswith(neuron_prefix)]
    for f in gt_files:
        src = os.path.join(gt_dir, f)
        dst = os.path.join(gt_subset_dir, f)
        shutil.copy2(src, dst)
        print(f"  Copied GT: {f}")


def estimate_nodes(zip_path, names, sample_size=50):
    """Estimate total node count by sampling a few SWC files."""
    import random
    sample = random.sample(names, min(sample_size, len(names)))
    total_nodes = 0
    with ZipFile(zip_path, "r") as zf:
        for name in sample:
            content = zf.read(name).decode("utf-8", errors="replace")
            n_lines = sum(1 for line in content.splitlines()
                          if line.strip() and not line.strip().startswith("#"))
            total_nodes += n_lines

    avg_nodes = total_nodes / len(sample) if sample else 0
    return int(avg_nodes * len(names))


def main():
    parser = argparse.ArgumentParser(description="Filter SWC fragments by proximity to GT")
    parser.add_argument("--max-dist", type=float, default=50,
                        help="Max distance (µm) from GT node to include a fragment")
    parser.add_argument("--max-fragments", type=int, default=2000,
                        help="Maximum number of fragments to extract")
    parser.add_argument("--gt-neuron", type=str, default=None,
                        help="Specific GT SWC file to use (basename)")
    args = parser.parse_args()

    # Validate args
    assert args.max_dist > 0, f"--max-dist must be positive, got {args.max_dist}"
    assert args.max_fragments > 0, f"--max-fragments must be positive, got {args.max_fragments}"

    # Verify data exists
    if not os.path.exists(SWC_ZIP):
        print(f"ERROR: SWC zip not found at {SWC_ZIP}", file=sys.stderr)
        print("Run download_data.py first.", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(GT_DIR):
        print(f"ERROR: GT directory not found at {GT_DIR}", file=sys.stderr)
        sys.exit(1)

    # Clean up previous subset if exists
    for d in [FRAGMENTS_DIR, GT_SUBSET_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # Select GT neuron
    if args.gt_neuron:
        gt_path = os.path.join(GT_DIR, args.gt_neuron)
    else:
        gt_path = find_gt_neuron(GT_DIR)

    neuron_prefix = os.path.basename(gt_path).split("-")[0]

    # Load ALL GT SWC files for this neuron (axon + dendrite + CA)
    print(f"\nLoading GT coordinates for {neuron_prefix}:")
    gt_coords = load_all_gt_coords(GT_DIR, neuron_prefix)
    assert gt_coords.ndim == 2 and gt_coords.shape[1] == 3, (
        f"GT coordinates have shape {gt_coords.shape}, expected (N, 3)"
    )
    if len(gt_coords) == 0:
        print(f"ERROR: No GT coordinates loaded for neuron prefix '{neuron_prefix}'.",
              file=sys.stderr)
        print(f"Check that GT files in {GT_DIR} start with '{neuron_prefix}'.",
              file=sys.stderr)
        sys.exit(1)
    assert np.isfinite(gt_coords).all(), "GT coordinates contain NaN or Inf"
    print(f"  Total GT nodes: {len(gt_coords):,}")
    print(f"  Absolute coordinate ranges:")
    for i, axis in enumerate(["X", "Y", "Z"]):
        print(f"    {axis}: [{gt_coords[:, i].min():.1f}, {gt_coords[:, i].max():.1f}]")

    # Build KD-tree for proximity queries
    print(f"\nBuilding KD-tree from {len(gt_coords):,} GT nodes...")
    gt_kdtree = cKDTree(gt_coords)

    # Filter fragments by proximity to GT
    print(f"\nScanning SWC zip for fragments within {args.max_dist} µm of GT...")
    matching = filter_fragments_by_proximity(
        SWC_ZIP, gt_kdtree, args.max_dist, args.max_fragments
    )

    if len(matching) == 0:
        print("ERROR: No matching fragments found! Try increasing --max-dist.")
        sys.exit(1)

    # Estimate node count
    est_nodes = estimate_nodes(SWC_ZIP, matching)
    print(f"\nEstimated total nodes: ~{est_nodes:,}")

    # Extract
    print(f"\nExtracting {len(matching)} fragments to {FRAGMENTS_DIR}")
    extract_fragments(SWC_ZIP, matching, FRAGMENTS_DIR)

    # Copy GT
    print(f"\nCopying GT neuron files to {GT_SUBSET_DIR}")
    copy_gt_for_subset(GT_DIR, GT_SUBSET_DIR, gt_path)

    # Summary
    n_extracted = len([f for f in os.listdir(FRAGMENTS_DIR) if f.endswith(".swc")])
    n_gt = len([f for f in os.listdir(GT_SUBSET_DIR) if f.endswith(".swc")])
    print(f"\n--- Summary ---")
    print(f"Fragments extracted: {n_extracted}")
    print(f"GT SWC files: {n_gt}")
    print(f"Estimated nodes: ~{est_nodes:,}")
    print(f"Max distance to GT: {args.max_dist} µm")
    print(f"Fragments dir: {FRAGMENTS_DIR}")
    print(f"GT dir: {GT_SUBSET_DIR}")


if __name__ == "__main__":
    main()
