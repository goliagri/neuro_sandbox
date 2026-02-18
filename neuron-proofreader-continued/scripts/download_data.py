"""
Download data for brain 653980 from the AIND open data S3 bucket.

Downloads:
  1. swcs-processed.zip (~1.4 GB) - predicted SWC fragments
  2. Ground truth SWC files from Complete_raw/ (~10 MB total)

Data is saved to ../neuro_data/653980/ relative to the repo root.
Uses boto3 directly (no aws CLI needed). Bucket is public (no credentials).
"""

import os
import sys

import boto3
from botocore import UNSIGNED
from botocore.config import Config as BotoConfig

# S3 config
BUCKET_NAME = "aind-open-data"
FUSION_PREFIX = "exaSPIM_653980_2023-08-10_20-08-29_fusion_2023-08-24"
RECON_PREFIX = "exaSPIM_653980_2023-08-10_20-08-29_reconstructions"

SWC_ZIP_KEY = f"{FUSION_PREFIX}/seg/swcs-processed.zip"
GT_PREFIX_KEY = f"{RECON_PREFIX}/Complete_raw/"

# Image path (not downloaded, accessed on-demand via TensorStore)
IMAGE_PATH = f"s3://{BUCKET_NAME}/{FUSION_PREFIX}/fused.zarr/0"

# Local paths
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(REPO_ROOT)), "neuro_data", "653980")
GT_DIR = os.path.join(DATA_DIR, "gt_swcs")


def get_s3_client():
    """Create an anonymous S3 client for the public bucket."""
    return boto3.client("s3", config=BotoConfig(signature_version=UNSIGNED))


def download_swc_zip(s3):
    """Download the predicted SWC fragments zip."""
    dest = os.path.join(DATA_DIR, "swcs-processed.zip")
    if os.path.exists(dest):
        size_mb = os.path.getsize(dest) / 1e6
        print(f"SWC zip already exists ({size_mb:.0f} MB), skipping download.")
        return dest

    print(f"\n--- Downloading predicted SWC fragments (~1.4 GB) ---")
    print(f"  s3://{BUCKET_NAME}/{SWC_ZIP_KEY} -> {dest}")

    # Get file size for progress
    head = s3.head_object(Bucket=BUCKET_NAME, Key=SWC_ZIP_KEY)
    total_bytes = head["ContentLength"]
    assert total_bytes > 0, f"S3 reports 0 bytes for {SWC_ZIP_KEY}"
    print(f"  Size: {total_bytes / 1e9:.2f} GB")

    downloaded = [0]
    def progress_callback(bytes_transferred):
        downloaded[0] += bytes_transferred
        pct = 100 * downloaded[0] / total_bytes
        mb = downloaded[0] / 1e6
        print(f"\r  Downloaded: {mb:.0f} MB ({pct:.1f}%)", end="", flush=True)

    s3.download_file(
        BUCKET_NAME, SWC_ZIP_KEY, dest, Callback=progress_callback
    )
    print()  # newline after progress
    actual_size = os.path.getsize(dest)
    assert actual_size == total_bytes, (
        f"Downloaded size {actual_size} != expected {total_bytes} bytes"
    )
    return dest


def download_gt_swcs(s3):
    """Download all ground truth SWC files."""
    if os.path.exists(GT_DIR) and len(os.listdir(GT_DIR)) > 0:
        n = len([f for f in os.listdir(GT_DIR) if f.endswith(".swc")])
        print(f"GT directory already has {n} SWC files, skipping download.")
        return

    os.makedirs(GT_DIR, exist_ok=True)
    print(f"\n--- Downloading ground truth SWC files ---")

    # List all .swc files under the GT prefix
    paginator = s3.get_paginator("list_objects_v2")
    swc_keys = []
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=GT_PREFIX_KEY):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".swc"):
                swc_keys.append(obj["Key"])

    assert len(swc_keys) > 0, (
        f"No SWC files found under s3://{BUCKET_NAME}/{GT_PREFIX_KEY}"
    )
    print(f"  Found {len(swc_keys)} SWC files")
    for key in swc_keys:
        filename = os.path.basename(key)
        dest = os.path.join(GT_DIR, filename)
        s3.download_file(BUCKET_NAME, key, dest)
        size_kb = os.path.getsize(dest) / 1e3
        assert size_kb > 0, f"Downloaded GT file is empty: {filename}"
        print(f"  Downloaded: {filename} ({size_kb:.0f} KB)")


def main():
    print(f"Data directory: {DATA_DIR}")
    print(f"Image (on-demand): {IMAGE_PATH}")
    os.makedirs(DATA_DIR, exist_ok=True)

    s3 = get_s3_client()
    download_swc_zip(s3)
    download_gt_swcs(s3)

    # Summary
    print("\n--- Summary ---")
    for root, dirs, files in os.walk(DATA_DIR):
        level = root.replace(DATA_DIR, "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 2 * (level + 1)
        for f in sorted(files)[:10]:
            size = os.path.getsize(os.path.join(root, f))
            print(f"{sub_indent}{f} ({size / 1e6:.1f} MB)")
        if len(files) > 10:
            print(f"{sub_indent}... and {len(files) - 10} more files")


if __name__ == "__main__":
    main()
