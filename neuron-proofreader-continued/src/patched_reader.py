"""
Monkey-patches for running neuron-proofreader on local data from S3.

Patches applied:
1. TensorStoreReader._load_image: Handle 5D OME-Zarr (strip singleton t,c)
   and reduce cache to save memory.
2. SwcReader.read: Fix missing method (read_from_paths calls self.read but
   the actual method is read_from_path).
3. ImageFeatureExtractor.__call__: Sequential processing to avoid OOM
   from concurrent large image patch reads.
4. GraphLoader.__call__: Sequential graph extraction instead of
   ProcessPoolExecutor with up to 512 workers (each spawned process
   copies parent memory, causing OOM on memory-constrained machines).

Must be called before any FragmentsDataset or InferencePipeline construction.
"""

import numpy as np

from neuron_proofreader.utils.img_util import TensorStoreReader
from neuron_proofreader.utils.swc_util import Reader as SwcReader

_SWC_DICT_REQUIRED_KEYS = {"id", "pid", "radius", "xyz", "swc_name"}


def patch_swc_reader():
    """
    Monkey-patch SwcReader to:
    1. Fix missing 'read' method (read_from_paths calls self.read but the
       actual method is read_from_path).
    2. Replace ProcessPoolExecutor in read_from_paths with sequential reads
       to avoid spawning worker processes that copy parent memory.
    """
    from collections import deque

    SwcReader.read = SwcReader.read_from_path

    def sequential_read_from_paths(self, swc_paths):
        swc_dicts = deque()
        for path in swc_paths:
            result = self.read_from_path(path)
            if result:
                assert isinstance(result, dict), (
                    f"SwcReader.read_from_path returned {type(result)}, expected dict"
                )
                assert _SWC_DICT_REQUIRED_KEYS.issubset(result.keys()), (
                    f"SWC dict missing keys: {_SWC_DICT_REQUIRED_KEYS - result.keys()}"
                )
                assert result["xyz"].ndim == 2 and result["xyz"].shape[1] == 3, (
                    f"SWC xyz has shape {result['xyz'].shape}, expected (N, 3)"
                )
                swc_dicts.append(result)
        return swc_dicts

    SwcReader.read_from_paths = sequential_read_from_paths


def patch_tensorstore_reader(cache_bytes=100_000_000):
    """
    Monkey-patch TensorStoreReader._load_image to:
    1. Handle 5D OME-Zarr arrays (strip singleton t,c dims)
    2. Reduce TensorStore cache to save memory (default 100 MB vs 1 GB)
    """
    import tensorstore as ts
    from neuron_proofreader.utils import util
    from neuron_proofreader.utils.img_util import get_storage_driver

    def patched_load(self):
        bucket_name, path = util.parse_cloud_path(self.img_path)
        storage_driver = get_storage_driver(self.img_path)
        self.img = ts.open(
            {
                "driver": self.driver,
                "kvstore": {
                    "driver": storage_driver,
                    "bucket": bucket_name,
                    "path": path,
                },
                "context": {
                    "cache_pool": {"total_bytes_limit": cache_bytes},
                },
                "recheck_cached_data": "open",
            }
        ).result()
        # Strip singleton t and c dimensions from 5D OME-Zarr -> 3D (z, y, x)
        if self.img.ndim == 5 and self.img.shape[0] == 1 and self.img.shape[1] == 1:
            self.img = self.img[0, 0]

        assert self.img.ndim == 3, (
            f"TensorStore image has {self.img.ndim} dimensions after stripping, "
            f"expected 3 (z, y, x). Original shape may be unsupported."
        )
        assert all(s > 0 for s in self.img.shape), (
            f"TensorStore image has zero-sized dimension: shape={self.img.shape}"
        )

    TensorStoreReader._load_image = patched_load


def patch_image_feature_extractor(cache=True, cache_dir=None, **kwargs):
    """
    Monkey-patch ImageFeatureExtractor.__call__ to process proposals
    sequentially instead of concurrently, with optional caching.

    The original uses unbounded ThreadPoolExecutor to read all proposal image
    patches concurrently. Each extractor holds a large intermediate crop
    (~200^3 float64 = 64 MB for img + mask), and all futures store their
    results in memory simultaneously. With 32 proposals this peaks at ~4 GB.
    Sequential processing keeps only one extractor in memory at a time.

    Parameters
    ----------
    cache : bool or str
        - True or "memory": In-memory dict cache (~14 MB/proposal).
        - "disk": Disk-based cache using numpy files in cache_dir.
          Much slower than memory but uses negligible RAM.
        - False: No caching (re-reads from S3 every epoch).
    cache_dir : str or None
        Directory for disk cache. Defaults to /tmp/claude/feature_cache.
    """
    import hashlib
    import os

    from neuron_proofreader.split_proofreading.split_feature_extraction import (
        ImageFeatureExtractor,
    )

    # Set up cache backend
    if cache == "disk":
        _cache_dir = cache_dir or "/tmp/claude/feature_cache"
        os.makedirs(_cache_dir, exist_ok=True)
        _mem_cache = None
    elif cache:
        _mem_cache = {}
        _cache_dir = None
    else:
        _mem_cache = None
        _cache_dir = None

    def _proposal_key(proposal):
        """Create a filesystem-safe key from a proposal frozenset."""
        key = "_".join(sorted(str(n) for n in proposal))
        return hashlib.md5(key.encode()).hexdigest()

    def _disk_load(proposal):
        """Load cached (patch, profile) from disk, or return None."""
        if _cache_dir is None:
            return None
        key = _proposal_key(proposal)
        patch_path = os.path.join(_cache_dir, f"{key}_patch.npy")
        profile_path = os.path.join(_cache_dir, f"{key}_profile.npy")
        if os.path.exists(patch_path) and os.path.exists(profile_path):
            return np.load(patch_path), np.load(profile_path)
        return None

    def _disk_save(proposal, patch, profile):
        """Save (patch, profile) to disk cache."""
        if _cache_dir is None:
            return
        key = _proposal_key(proposal)
        np.save(os.path.join(_cache_dir, f"{key}_patch.npy"), patch)
        np.save(os.path.join(_cache_dir, f"{key}_profile.npy"), profile)

    def _extract_proposal(self, subgraph, proposal):
        """Extract patch and profile for a single proposal (full S3 read + compute)."""
        extractor = self.init_extractor(subgraph, proposal)

        profile = extractor.get_intensity_profile()
        assert isinstance(profile, np.ndarray), (
            f"get_intensity_profile returned {type(profile)}, expected ndarray"
        )
        assert profile.ndim == 1, (
            f"Intensity profile has shape {profile.shape}, expected 1D"
        )

        patch = extractor.get_input_patch()
        assert isinstance(patch, np.ndarray), (
            f"get_input_patch returned {type(patch)}, expected ndarray"
        )
        expected_patch_shape = (2,) + tuple(self.patch_shape)
        assert patch.shape == expected_patch_shape, (
            f"Patch shape {patch.shape} != expected {expected_patch_shape}"
        )
        assert np.isfinite(patch).all(), "Patch contains NaN or Inf values"

        return patch, profile

    def patched_call(self, subgraph, features):
        patches, profiles = dict(), dict()
        cache_hits = 0
        for proposal in subgraph.proposals:
            # Try memory cache
            if _mem_cache is not None and proposal in _mem_cache:
                patch, profile = _mem_cache[proposal]
                cache_hits += 1
            else:
                # Try disk cache
                cached = _disk_load(proposal)
                if cached is not None:
                    patch, profile = cached
                    cache_hits += 1
                else:
                    # Full extraction from S3
                    patch, profile = _extract_proposal(self, subgraph, proposal)
                    _disk_save(proposal, patch, profile)

                # Store in memory cache
                if _mem_cache is not None:
                    _mem_cache[proposal] = (patch, profile)

            profiles[proposal] = profile
            patches[proposal] = patch

        if cache_hits > 0:
            total = len(subgraph.proposals)
            backend = "memory" if _mem_cache is not None else "disk"
            print(f"  [cache:{backend}] {cache_hits}/{total} proposals from cache")

        features.set_features(patches, "proposal_patches")
        features.integrate_proposal_profiles(profiles)

    ImageFeatureExtractor.__call__ = patched_call

    def get_cache_info():
        """Return cache stats: (n_entries, total_bytes)."""
        if _mem_cache is not None:
            n = len(_mem_cache)
            total_bytes = sum(
                p.nbytes + pr.nbytes for p, pr in _mem_cache.values()
            )
            return n, total_bytes
        if _cache_dir is not None:
            files = [f for f in os.listdir(_cache_dir) if f.endswith(".npy")]
            n = len(files) // 2  # patch + profile per proposal
            total_bytes = sum(
                os.path.getsize(os.path.join(_cache_dir, f)) for f in files
            )
            return n, total_bytes
        return 0, 0

    def clear_cache():
        """Clear the feature cache."""
        if _mem_cache is not None:
            _mem_cache.clear()
        if _cache_dir is not None:
            import shutil
            shutil.rmtree(_cache_dir, ignore_errors=True)
            os.makedirs(_cache_dir, exist_ok=True)

    # Expose cache controls
    ImageFeatureExtractor._feature_cache_info = staticmethod(get_cache_info)
    ImageFeatureExtractor._feature_cache_clear = staticmethod(clear_cache)


def patch_graph_loader():
    """
    Monkey-patch GraphLoader.__call__ to extract graphs sequentially.

    The original uses ProcessPoolExecutor with up to 512 concurrent workers
    (graph_util.py:168-171). Each spawned process copies the parent's memory
    space, which causes OOM on memory-constrained machines. Sequential
    processing uses only the parent process's memory.
    """
    from collections import deque
    from tqdm import tqdm
    from neuron_proofreader.utils.graph_util import GraphLoader

    def patched_call(self, fragments_pointer):
        swc_dicts = self.swc_reader(fragments_pointer)

        desc = "Extract Graphs"
        pbar = tqdm(total=len(swc_dicts), desc=desc) if self.verbose else None
        irreducibles = deque()
        high_risk_cnt = 0

        while swc_dicts:
            extraction = self.extract(swc_dicts.pop())
            assert isinstance(extraction, tuple) and len(extraction) == 2, (
                f"GraphLoader.extract returned {type(extraction)}, expected (deque, int)"
            )
            result, cnt = extraction
            assert isinstance(cnt, int) and cnt >= 0, (
                f"high_risk_cnt is {cnt!r}, expected non-negative int"
            )
            high_risk_cnt += cnt
            irreducibles.extend(result)
            if pbar:
                pbar.update(1)

        if self.verbose and high_risk_cnt > 0:
            print("# High Risk Merges Detected:", high_risk_cnt)
        return irreducibles

    GraphLoader.__call__ = patched_call
