# Model card: HybridRestoreNet

## Status

Architecture and training code only. No weights or measured performance are released.
Research use only.

## Inputs and outputs

Input `[B,3,D,H,W]`: normalized HU, artifact mask, known-data confidence. Output:
bounded residual-corrected HU representation, residual, artifact logit, and log
variance. Full-volume inference uses overlapping Gaussian-weighted patches.

## Intended use

Retrospective research on artifact correction and denoising in adult head-and-neck CT.
Not intended for independent diagnosis, treatment planning, dose calculation, pediatric
use, or replacement of raw/vendor-reconstructed images.

## Main risks

Hallucinated anatomy, HU bias, oversmoothing, mask failure, synthetic-to-real domain
shift, scanner/site bias, poor uncertainty calibration, and changes outside artifacts.
Known-region identity loss reduces but does not eliminate these risks.

## Efficiency

Depthwise spatial mixing, 1x1 gated channel projection, and bottleneck-only factorized
axial kernels avoid quadratic attention. Mixed precision, EMA, DDP, and gradient
accumulation are supported. Measure full-volume latency and memory on the exact release
hardware; parameter count alone is not a deployment benchmark.

