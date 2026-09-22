# Event Camera Foundation Model (ECFM) Design

This document describes a small, flexible transformer-based foundation model for
event camera streams represented as (x, y, t, p) tuples. It targets fast
iteration with small models and supports multi-scale spatio-temporal regions.

## Goals

- Tokenize event streams into histogram patches over arbitrary regions.
- Embed region position/size explicitly for flexible masking and scaling.
- Train with MAE-style masked reconstruction over sampled regions, which may overlap.
- Support multi-scale (dx, dy, dt) to cover short and long integration times.
- Keep a simple, modular PyTorch codebase that scales later.

## Data Representation

### Event Stream

Each event is `(x, y, t, p)` where `p` is polarity. Input events are assumed to
use pixel coordinates, normalized sequence time, and polarity IDs 0 or 1. The
histogram builders do not require time-sorted input.

### Region Definition

Regions are defined by `(x, y, t, dx, dy, dt)` with an aggregation plane:

- `xy`: integrate over the region's time interval to produce a spatial histogram.
- `xt`: integrate over y to produce x-vs-time histograms.
- `yt`: integrate over x to produce y-vs-time histograms.
- `xy_p45`, `xy_m45`, `yt_p45`, and `yt_m45`: supported rotated projections.

Region selection is random or grid-based. Random regions independently sample
their spatial size, duration, origin, and plane and may overlap. Spatial and
temporal origins are constrained so the complete region remains in bounds;
full-frame and full-duration regions start at zero.

### Histogram Patch

1) Aggregate event counts into a histogram H over the chosen plane.
2) Normalize per polarity channel using the configured normalization mode
   (`region_max`, `region_sum`, `region_mean`, `none`, or a fixed divider).
3) Compute a separate log event-rate reconstruction target from the total count.
4) Resize the normalized histogram into a fixed patch size (e.g., 16x16 or 32x32).
5) Create a two-channel patch tensor for negative and positive polarity.

## Tokenization

Each region yields one token with:

- Patch: resized two-channel polarity histogram.
- Metadata: normalized position/size, normalized time, corresponding times in
  seconds, and total sequence duration.
- A learned embedding for the aggregation-plane ID.

Spatial position and size are normalized by the configured image bounds. Start
time and duration are normalized by sequence duration; their values in seconds
are also included in metadata.

Metadata is injected with an MLP and added to the patch embedding together with
a learned plane embedding.

The separate density target is
`log1p(total_events / (dx * dy * dt_seconds))`. It is not part of the token
metadata and only contributes to training when `count_loss_weight > 0`.

## Model Architecture

- Patch encoder: two convolution/ReLU blocks followed by a linear projection.
- Metadata encoder: MLP for region attributes.
- Token vector: sum patch, metadata, and plane embeddings in `d_model`.
- Transformer encoder with optional learned relative attention bias.
- Reconstruction heads: predict the masked patch and optional log event rate.

### Relative Attention Bias

For query region `i` and key region `j`, the bias network receives six pairwise
features:

`(x_i-x_j, y_i-y_j, t_i-t_j, log(dx_i/dx_j), log(dy_i/dy_j), log(dt_i/dt_j))`

A small MLP maps these features to one additive bias per attention head. The
bias is recomputed from region metadata for each batch and shared by all
encoder layers. Configurations using relative bias normally set
`use_pos_embedding: false`, making region geometry rather than list order the
source of positional information.

Variable region counts are padded to the largest configured count. The dataset
returns a `valid_mask`; padded keys are excluded from encoder and decoder
attention, and padded model outputs are zeroed.

Small-model starting point:

- `d_model`: 192
- `n_layers`: 4
- `n_heads`: 3
- `mlp_ratio`: 2

## Masking Strategy (MAE-style)

Randomly select valid regions and hide their patch content. The encoder still
receives a learned mask token together with that region's metadata and plane
identity. This is a masked-token autoencoder, not the original MAE design where
the encoder receives visible tokens only. The decoder reconstructs masked
tokens using:

- Patch reconstruction loss: summed MSE, divided by the number of masked
  regions. An optional Gaussian blur can be applied to prediction and target.
- Optional density loss: L1 between `log1p(softplus(prediction))` and the log
  event-density target.

Masking is uniform over valid tokens. For `V` valid regions, the implementation
masks `int(V * mask_ratio)` regions. Padding is never selected as a target.

## Multi-Scale Strategy

Represent multiple region sizes within the same sequence:

- Sample a mix of dx/dy/dt scales per batch.
- Encode scale explicitly in metadata.
- Configure scales as absolute pixels or fractions of image dimensions.
- Optionally sample the token count from `num_regions_choices`; samples are
  padded to the largest configured count.

## Augmentations

The currently wired training augmentation is arbitrary spatial rotation of
event coordinates using nearest-neighbor rounding. It is enabled with
`augmentations: ["rotate"]` and `rotation_max_deg > 0`. Rotation preserves time
and polarity and clips coordinates to image bounds.

## Datasets

The pretraining loader currently supports synthetic events, THU-EACT, and
DVS-Lip. Dataset configuration supplies sensor resolution and timestamp units;
each loaded sequence is normalized to `[0, 1]` in time before region sampling.

## Evaluation and Diagnostics

- Reconstruction error across masked regions and scales.
- Saved ground-truth/prediction images for a fixed masked sample across epochs.
- Linear probing and supervised fine-tuning for supported classification paths.

Optical flow, representation consistency across planes, and a contrastive
pretraining term remain possible future work; they are not implemented in the
region-MAE training path.

## Implemented Components

1) Event-to-patch pipelines for spatial, temporal, and rotated projections.
2) Random and grid region sampling with fixed or variable token counts.
3) Masked region-token autoencoder with learned relative attention bias.
4) Synthetic, THU-EACT, and DVS-Lip dataset paths.
5) Rotation augmentation, token caching, reconstruction dumps, linear probing,
   and fine-tuning entrypoints.

## Open Questions

- Best loss weighting between patch reconstruction and event density.
- Patch size vs region size tradeoffs.
- How to mix plane types per batch for stable training.
