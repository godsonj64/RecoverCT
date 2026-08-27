# Research rationale (reviewed 2026-08-27)

## Problem formulation

Metal artifact reduction is not ordinary image completion. Dental metal causes beam
hardening, photon starvation, scatter, partial-volume effects, and nonlinear errors in
projection measurements. Streaks can cross tissue far from the implant. A model can
score well perceptually while changing clinically important HU or inventing anatomy.
The task is therefore constrained residual estimation in HU with uncertainty and
dosimetric validation, not unconstrained generative inpainting.

## Evidence translated into the baseline

1. **Dual-domain priors help.** DCDiff and related 3D CBCT work condition on image and
   projection information. They are important comparators. TCIA planning collections
   do not normally expose raw sinograms, so the baseline uses forward-projection
   corruption during training and remains image-domain at deployment. A raw-data site
   should add a sinogram branch and data-consistency operator.
2. **Loss must focus on affected tissue.** The 2025 head-and-neck MAR work reports a
   masked criterion and dose recalculation. `RestorationLoss` combines affected-region
   L1, global fidelity, known-region identity, 3D gradients, local structural
   similarity, artifact-mask supervision, and heteroscedastic error.
3. **Long-range context need not mean quadratic attention.** DASMamba and later CT
   state-space models support linear-complexity global processing. For a portable first
   implementation, the repository uses large factorized depthwise axial kernels only
   at the bottleneck. A true selective state-space block is an ablation, not an
   unverified dependency in the baseline.
4. **Qwen is an efficiency analogy, not a medical backbone.** Qwen3-Next mixes cheap
   gated linear processing with occasional expensive attention and sparsely activates
   capacity. Here that maps to gated depthwise blocks, bottleneck-only global mixing,
   and multi-head outputs. Sparse MoE was rejected for the small 3D model because its
   routing/communication overhead can exceed its savings and complicates validation.
5. **Diffusion is a research comparator.** It can represent ambiguous corrections but
   costs multiple passes and increases hallucination/variance concerns. Evaluate it as
   an offline teacher or ensemble, not as the sole production baseline.

## Proposed ablation ladder

Run all experiments with identical patient splits and synthetic seeds:

1. 3D residual U-Net;
2. + depthwise gated blocks;
3. + axial bottleneck mixer;
4. + mask and uncertainty heads;
5. + Radon corruption versus analytic streaks;
6. optional Mamba/SSM bottleneck;
7. optional dual-domain branch when raw projections exist;
8. optional diffusion teacher distilled into the deterministic student.

Report parameters, peak memory, training energy/GPU-hours, latency per full volume,
and every image/dose endpoint. “State of the art” must not be claimed from one dataset
or synthetic artifacts alone.

## Data gaps

The public collections provide real patients but not exact pre-/post-artifact pairs.
Strong clinical evidence needs prospective or retrospective institutional pairs such as:

- same-session reconstruction with and without a validated vendor MAR method;
- registered kVCT/MVCT with residual registration error characterized;
- anthropomorphic phantom scans with known inserts and ground truth;
- raw projection data with metal-free and metal-insert acquisitions;
- multi-vendor, multi-kVp, multi-kernel, and multi-institution external test sets.

Disease-free and artifact-free are separate labels. A cancer patient's planning CT can
still provide clean normal-tissue patches outside tumor and artifacts if masks are
reviewed. Conversely, high visual quality does not prove correct HU.

