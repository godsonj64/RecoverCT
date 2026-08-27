# Clinical validation and release gate

This is a protocol template, not a clinical approval.

## Locked evaluation cohorts

- Split by patient and institution, never by slice or patch.
- Hold out at least one scanner vendor/site and all phantom acquisitions.
- Stratify dental material, implant count/size, kVp, mAs, reconstruction kernel, slice
  thickness, contrast, tumor site, body habitus, and artifact severity.
- Include artifact-free scans to quantify harmful changes and false-positive behavior.
- Freeze weights, thresholds, normalization, and mask process before testing.

## Required endpoints

Image-domain:

- MAE, bias, RMSE, and 95th-percentile absolute error in HU;
- results inside artifact, metal-adjacent, and known clean regions;
- air, soft tissue, cortical/dense bone, and site-specific regions of interest;
- 3D SSIM/PSNR as secondary—not clinical—metrics;
- edge displacement, small-structure preservation, and artifact index;
- failure rate, uncertainty calibration, selective-risk curve, and OOD sensitivity.

Radiotherapy:

- recalculate the unchanged approved plan on source, corrected, and reference CT using
  the same TPS version, dose algorithm, grid, calibration curve, and structures;
- compare target D98/D95/D50/D2 and OAR Dmean/Dmax plus site-specific constraints;
- 3D gamma and spatial dose difference near dental metal and interfaces;
- contouring inter/intra-observer study where anatomy was obscured;
- anthropomorphic phantom end-to-end measurement with dental inserts.

All equivalence/non-inferiority margins, gamma criteria, exclusions, sample size, and
statistical analysis must be set prospectively by the institution's qualified medical
physicists and clinicians. Do not copy a literature threshold without local rationale.

## Fail-closed deployment behavior

- Preserve and display the original CT next to the output.
- Overlay the changed-voxel map, input artifact mask, and calibrated uncertainty.
- Reject unsupported orientation, spacing, modality, reconstruction kernel, kVp,
  truncated anatomy, or HU range; do not silently rescale.
- Reject a case when mask confidence or uncertainty exceeds validated limits.
- Record model checksum, source SOP/series UIDs, preprocessing, runtime, operator, and
  software/hardware versions in an immutable audit trail.
- Require explicit clinician/physicist review; prohibit automatic TPS overwrite.
- Maintain rollback, drift monitoring, incident review, and periodic revalidation.

## Governance evidence before any clinical use

- intended-use statement and hazard analysis;
- data provenance, consent/access compliance, and demographic/site bias analysis;
- software lifecycle, cybersecurity, change control, and locked dependency artifacts;
- independent code review and reproducible build;
- local commissioning plus external validation;
- regulatory determination in each intended jurisdiction;
- compliance with the joint ESTRO/AAPM guidance for AI in radiation therapy.

