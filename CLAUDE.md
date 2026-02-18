# neuro_sandbox

## Data
Project data is stored at `../neuro_data`.

## neuron-proofreader-main
Allen Institute repo (by Anna Grim) for automated proofreading of neuron skeleton reconstructions from exaSPIM whole-brain light-sheet microscopy.

**Two independent pipelines:**
- **Split proofreading** (main functionality): Reconnects over-fragmented neuron skeleton pieces. Pipeline: graph construction → proposal generation → GNN classification → merge accepted proposals.
- **Merge proofreading** (incomplete/in-progress): Detects where two neurons were erroneously joined. Has stub methods.

**Key inputs:**
- SWC files (neuron morphology skeletons — text format with id, type, x, y, z, radius, parent columns)
- Volumetric imagery via TensorStore (Zarr/N5/precomputed from GCS or S3)
- Trained model weights

**Architecture:**
- `SkeletonGraph` / `ProposalGraph` — NetworkX-based graph representations with irreducible compression (only leaf + branching nodes kept)
- `ImageFeatureExtractor` — reads 96³ patches from cloud storage per proposal
- `VisionHGAT` — heterogeneous graph attention network for proposal classification
- No CLI — library-only, programmatic invocation

**Public compatible data:** `s3://aind-open-data` has exaSPIM datasets with fused.zarr imagery and SWC fragments. Authors' private data (`gs://allen-nd-goog`) is not accessible. Authors process data in spatial blocks, not whole brains at once.
