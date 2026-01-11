# Event Camera Foundation Model (ECFM) Design Draft

This document proposes a small, flexible transformer-based foundation model for
event camera streams represented as (x, y, t, p) tuples. It targets fast
iteration with small models and supports multi-scale spatio-temporal regions.

## Goals
- Tokenize event streams into histogram patches over arbitrary regions.
- Embed region position/size explicitly for flexible masking and scaling.
- Train with MAE-style masked reconstruction over non-overlapping regions.
- Support multi-scale (dx, dy, dt) to cover short and long integration times.
- Keep a simple, modular PyTorch codebase that scales later.

## Data Representation
### Event Stream
Each event is `(x, y, t, p)` where `p` is polarity. Input events are assumed to
be sorted by time or can be sorted at load time.

### Region Definition
Regions are defined by `(x, y, t, dx, dy, dt)` with an aggregation plane:
- `xy`: integrate over time bins to produce spatial histograms per region.
- `xt`: integrate over y to produce x-vs-time histograms.
- `yt`: integrate over x to produce y-vs-time histograms.

Region selection is flexible and can be random, grid-based, or dataset-specific.

### Histogram Patch
1) Aggregate event counts into a histogram H over the chosen plane.
2) Max-normalize: `Hn = H / max(H, eps)`.
3) Preserve total events `E = sum(H)` to re-scale if needed.
4) Resize `Hn` into a fixed patch size (e.g., 16x16 or 32x32).
5) Create a patch tensor and concatenate or fuse `E` and metadata.

## Tokenization
Each region yields one token with:
- Patch: resized histogram (1 or 2 channels for polarity).
- Scalar metadata: total events E, region position/size, aggregation plane.

Two position/size options:
- Absolute: normalize x,y,t,dx,dy,dt by global bounds.
- Relative: normalize by local sequence bounds or by a parent region.

Metadata can be injected by:
- A small MLP appended to patch embedding.
- Adding a learned embedding per plane type (xy/xt/yt).
- Concatenating positional encodings to the token vector.

## Model Architecture
- Patch encoder: small CNN or linear projection on flattened patch.
- Metadata encoder: MLP for region attributes.
- Token vector: concat/merge patch + metadata into `d_model`.
- Transformer encoder (ViT-like).
- Reconstruction head: predict masked patch + total events.

Small-model starting point:
- `d_model`: 192
- `n_layers`: 4
- `n_heads`: 3
- `mlp_ratio`: 2

## Masking Strategy (MAE)
Mask any set of non-overlapping regions in space-time.
Given visible tokens, reconstruct masked tokens:
- Patch reconstruction loss: L1 or MSE on normalized patches.
- Event count loss: L1 on total events E.
Optionally use a contrastive term to keep representations stable across scales.

Mask sampling variants:
- Random subsets across planes (xy/xt/yt).
- Scale-aware masks (mask larger regions more often).
- Coverage constraints to avoid leaving large gaps.

## Multi-Scale Strategy
Represent multiple region sizes within the same sequence:
- Sample a mix of dx/dy/dt scales per batch.
- Encode scale explicitly in metadata.
- Optionally add a scale token per scale group.

## Augmentations
All augmentations operate on events:
- Spatial rotations (90/180/270 or arbitrary with interpolation).
- Spatial flips.
- Time warp: non-linear t->t' with monotonic mapping.
- Optional polarity flips.

Temporal jitter can be optional; scale diversity may be sufficient.

## Datasets
Support heterogeneous datasets:
- Stationary vs moving sensors.
- Varying resolutions and time spans.
- Normalize coordinates to dataset-specific bounds.

## Evaluation Ideas
- Reconstruction error across masked regions and scales.
- Downstream tasks: classification, optical flow, detection.
- Probe representation consistency across planes.

## Implementation Plan
1) Build event-to-patch pipeline for xy/xt/yt.
2) Implement dataset class to sample regions and mask sets.
3) Implement tiny transformer + MAE training loop.
4) Add augmentations and multi-dataset mix.

## Open Questions
- Best loss weighting between patch vs event count.
- Patch size vs region size tradeoffs.
- How to mix plane types per batch for stable training.

