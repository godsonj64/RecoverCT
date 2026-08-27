# CT-Restore

Compact, volumetric CT artifact inpainting and denoising for **head-and-neck
radiotherapy research**. The repository includes TCIA discovery/download tools,
DICOM-to-NIfTI preprocessing, patient-level splits, explicit quality-control gates,
physics-informed artifact synthesis, staged training, sliding-window inference,
uncertainty output, and HU-stratified evaluation.

> **Safety status:** research use only. There are no released trained weights and no
> claim of clinical performance. Output must not be imported into a treatment planning
> system or used for diagnosis, contouring, electron-density assignment, or dose
> calculation until it has passed local medical-device governance, independent
> commissioning, phantom testing, multi-site validation, and qualified medical
> physicist review. Always retain the source DICOM and vendor reconstruction.

## Colab and RunPod quick start

The repository includes a complete portable notebook:
[CT_Restore_Colab_RunPod_End_to_End.ipynb](notebooks/CT_Restore_Colab_RunPod_End_to_End.ipynb).
It performs environment setup, CUDA detection, a guarded TCIA sample download,
preprocessing, one smoke-training epoch, inference, and visual inspection.

After cloning on RunPod or another Linux GPU notebook:

```bash
git clone https://github.com/godsonj64/RecoverCT.git ct-restore
cd ct-restore
export CT_RESTORE_DATA_ROOT=/workspace/ct_restore_data
bash scripts/bootstrap_notebook.sh

# Review the linked TCIA terms first. This downloads one series by default.
python scripts/run_notebook_pipeline.py \
  --data-root "$CT_RESTORE_DATA_ROOT" \
  --limit 1 --epochs 1 --accept-data-terms
```

For Google Colab, upload/open the notebook, set `REPO_URL` to the published fork,
select a GPU runtime, review the TCIA terms, and run cells in order. Set
`USE_GOOGLE_DRIVE=True` if persistence matters more than I/O speed. The notebook will
not download data until `ACCEPT_TCIA_DATA_TERMS=True` is explicitly set.

The automated path is a pipeline smoke test, not scientific training: it bypasses the
manual image-review gate, changes split assignment only in a manifest labeled
`smoke_only_split_override`, and produces weights that must not be promoted. Full
experiments require reviewed manifests and the staged configs below. See the detailed
[notebook runtime guide](docs/notebook_runtimes.md).

## Automatic hardware acceleration

`ct-restore doctor` detects local, Colab, or RunPod execution and reports the GPU,
VRAM, CUDA version, compute capability, precision, worker count, and recommended 3D
patch. `hardware.auto_tune: true` applies conservative hardware-specific settings and
records them in `resolved_config.json` and the checkpoint.

When CUDA is available, training automatically uses:

- BF16 on supported Ampere-or-newer hardware, otherwise FP16 with gradient scaling;
- TF32 tensor cores for eligible FP32 convolution/matmul operations;
- cuDNN benchmarking for fixed patch shapes;
- 3D channels-last layout;
- fused AdamW when supported, with an automatic ordinary-AdamW fallback;
- pinned memory, non-blocking transfers, persistent workers, and prefetching;
- multi-GPU DDP when launched with `torchrun`.

CPU and Apple MPS remain functional fallbacks with smaller recommended patches. The
optional `hardware.compile` switch is off by default because compilation support and
benefit vary by GPU/runtime; if explicitly requested on an unsupported environment,
the trainer falls back to eager mode. Inspect or materialize a resolved configuration:

```bash
ct-restore doctor
ct-restore configure configs/notebook.yaml /workspace/notebook_resolved.yaml
```

## Why this design

`HybridRestoreNet` is a 3D residual U-Net with depthwise gated CNN blocks for local
detail and factorized axial mixers for long-range streak context. It has three heads:
corrected CT, refined artifact likelihood, and voxel-level heteroscedastic uncertainty.
The input contains calibrated CT, a reviewed/suspected artifact mask, and its known-data
confidence map.

The efficiency ideas are analogous to recent Qwen-family design principles—gated
feed-forward paths and a mixture of cheap local processing with sparse global
processing—but no language model is embedded in the image pipeline. Depthwise 3D
convolution and bottleneck-only axial mixing keep cost linear in voxel count. The
default network is about 1.77 million parameters (verify on your installation with
`ct-restore inspect-model`).

Diffusion was not selected for the deployable baseline: it is expensive, stochastic,
and can generate plausible but incorrect anatomy. Dual-domain sinogram/image models
are a strong research direction, but public radiotherapy DICOM collections generally do
not contain the raw projections needed at inference. See [research rationale](docs/research.md).

## Data strategy

| Role | Collection | Why | Access/size |
|---|---|---|---|
| Primary fine-tuning | `HNC-IMRT-70-33` | 211 head-and-neck cases with CT, RTSTRUCT, RTDOSE, and RTPLAN; directly relevant to planning. **Not served by the anonymous NBIA API** — requires an NBIA login or the official Data Retriever | TCIA, about 23.27 GB |
| External validation | `HEAD-NECK-PET-CT` | 298 subjects from four institutions; planning CT and RT objects | TCIA controlled-access conditions may apply, about 72.46 GB |
| Optional controlled cohort | `HNSCC` / `HEAD-NECK-CT-ATLAS` | Larger RT archive | controlled access, roughly 100–310 GB depending subset |

There is no scientifically defensible public cohort of “healthy, disease-free” people
who received planning CT solely for model training. CT uses ionizing radiation, and the
selected cancer collections contain disease. The pipeline instead:

1. selects planning-like axial CT series;
2. rejects localizers, thick slices, insufficient coverage, non-finite values, and
   obvious high-density metal;
3. writes every QC result to a manifest;
4. **requires a human to set `manual_approved=true`** before normal training;
5. keeps patient IDs entirely within one deterministic train/validation/test split.

Auto-QC cannot prove that a CT is artifact-free or that a patch is tumor-free. If tumor
exclusion is required for a specific experiment, rasterize the RTSTRUCT/SEG and add an
exclusion mask; do not infer health from intensity alone.

## Setup

Recommended: Linux, Python 3.10–3.14, CUDA-capable GPU with at least 24 GB memory for
the default patch, and an external data volume with at least 150 GB free if using both
recommended collections.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,dev]'
ct-restore inspect-model
```

The current checkout intentionally contains no patient data, pretrained weights, or
TCIA credentials.

### Import from Python or a notebook

After `pip install -e .` (clone) or `pip install dist/ct_restore-0.1.0-py3-none-any.whl` (built wheel), the model and simulator
are ordinary Python imports:

```python
import numpy as np
import torch

from ct_restore.artifacts import ArtifactSimulator
from ct_restore.hardware import detect_hardware
from ct_restore.models import HybridRestoreNet

profile = detect_hardware()
model = HybridRestoreNet(base_channels=profile.base_channels).to(profile.device)
clean_hu = np.zeros((32, 96, 96), dtype=np.float32)
synthetic_pair = ArtifactSimulator(seed=2026)(clean_hu)
print(profile.to_dict(), synthetic_pair.metadata)
```

The full download/preprocess workflow intentionally remains in the cloned repository
because it includes configs, notebook assets, scripts, data-use prompts, and audit
documentation that should travel together.

## End-to-end workflow

### 1. Query before downloading

The TCIA command defaults to a dry run. Review the generated series metadata, TCIA's
current access policy, data-use terms, and storage estimate before adding `--download`.

> **Access note:** `HNC-IMRT-70-33` and `HEAD-NECK-PET-CT` return zero series over the
> anonymous NBIA API, so the plain commands below retrieve nothing. Use one of the two
> authenticated routes described next. The Colab/RunPod notebook defaults to a public
> collection for its smoke run, which exercises the pipeline but is not head-and-neck
> planning data.

#### Restricted collections

Credentials are read from `NBIA_USERNAME` / `NBIA_PASSWORD` or an interactive hidden
prompt, exchanged once for an OAuth token, and never written to disk. Do not pass a
password as a command-line argument.

```bash
# Route 1: authenticate, then query and download as usual.
export NBIA_USERNAME=your_account
read -rs NBIA_PASSWORD && export NBIA_PASSWORD
ct-restore tcia --collection HNC-IMRT-70-33 --login \
  --output-dir /external/ct_restore/raw --download --limit 1

# Route 2: export a .tcia manifest from the TCIA Data Retriever, then fetch it.
ct-restore fetch-manifest manifest-1699999999999.tcia \
  --output-dir /external/ct_restore/raw --login --limit 1
```

Route 2 is the more reliable of the two, because it works for any collection the web
interface lets you basket, regardless of API exposure.

```bash
ct-restore collections
ct-restore tcia \
  --collection HNC-IMRT-70-33 \
  --output-dir /external/ct_restore/raw \
  --manifest data/manifests/hnc_imrt_series.csv

# Small retrieval test after review
ct-restore tcia --collection HNC-IMRT-70-33 \
  --output-dir /external/ct_restore/raw --download --limit 1

# Full retrieval: remove --limit only when sufficient storage is mounted.
```

### 2. Preprocess and approve

```bash
ct-restore preprocess \
  /external/ct_restore/raw/HNC-IMRT-70-33 \
  /external/ct_restore/processed/HNC-IMRT-70-33 \
  --manifest data/manifests/head_neck.csv
```

Review image orientation, coverage, HU calibration, metal, motion, truncation, contrast,
tumor/structure masks, and reconstruction kernel. Then edit the CSV's
`manual_approved` field. Preprocessing preserves floating-point HU, uses linear
resampling to `(z,y,x)=(1.5,1.0,1.0) mm`, and records the original QC evidence.

### 3. Staged training

Stage 1 uses diverse, visually reviewed clinical CT as calibrated volumetric grayscale
and synthesizes noise plus metal corruption. Stage 2 starts from that checkpoint and
uses reviewed head-and-neck planning CT. A true paired real-artifact stage can be added
by extending the manifest with institutionally acquired, registered vendor-MAR or
MVCT targets; never label an unpaired image as ground truth.

```bash
ct-restore train configs/pretrain.yaml
ct-restore train configs/finetune.yaml
# Only when reviewed, registered real pairs are available:
ct-restore train configs/paired_real.yaml

# Multi-GPU
torchrun --standalone --nproc-per-node=4 -m ct_restore.cli train configs/finetune.yaml
```

The corruption simulator inserts correlated 3D dental implant shapes and uses a
slice-wise Radon/FBP approximation of beam hardening, photon starvation, detector
variation, and low-dose noise. It is augmentation—not a validated scanner simulator.

### 4. Inference

Prefer a reviewed artifact mask. Without one, the CLI uses a deliberately conservative
high-density-metal heuristic that will miss some nonmetal artifacts.

```bash
ct-restore infer scan.nii.gz checkpoints/finetune/best.pt restored.nii.gz \
  --mask reviewed_artifact_mask.nii.gz
```

Inference writes the corrected volume, an **uncalibrated** uncertainty map, and a JSON
provenance sidecar. Do not use low uncertainty as proof of correctness.

### 5. Evaluation

```bash
ct-restore evaluate restored.nii.gz clean_reference.nii.gz \
  --artifact-mask artifact_mask.nii.gz --output metrics.json
```

Image metrics are necessary but insufficient. Clinical evaluation must include tissue-
specific HU bias, contour consistency, end-to-end phantom tests, TPS dose recalculation,
DVH endpoints, and gamma analysis with limits prospectively set by local physicists.
See the [clinical validation plan](docs/clinical_validation.md).

## Reproducibility and controls

- Patient-level split is a stable SHA-256 mapping; no slice leakage.
- Resolved configs, optimizer, scheduler, EMA weights, epoch, and validation loss are
  stored in each checkpoint.
- Automatic mixed precision, gradient accumulation, gradient clipping, EMA, and DDP
  are supported.
- TCIA download is opt-in, retrying/resumable through the NBIA API, and metadata is retained.
- Raw/processed DICOM, NIfTI, model weights, outputs, and manifests are git-ignored.
- The manual QC gate can only be bypassed with the visibly named
  `--allow-unreviewed` option for smoke tests.

## Repository map

```text
configs/                  staged experiment configs
docs/                     research, data, model, and validation documentation
notebooks/                Colab/RunPod end-to-end notebook
scripts/                  portable bootstrap and guarded smoke pipeline
src/ct_restore/artifacts/ physics-informed synthetic corruption
src/ct_restore/data/      TCIA, DICOM preprocessing, split and patch dataset
src/ct_restore/models/    compact hybrid 3D network
src/ct_restore/           losses, training, inference, metrics, CLI
tests/                    unit and smoke tests
```

## Current limitations

- No pretrained model or benchmark result is included.
- TCIA does not provide clean/metal-corrupted pairs or raw projections for these
  planning collections.
- Synthetic-to-real shift remains the central research risk.
- The uncertainty head is not calibrated until evaluated on held-out real artifacts.
- Contrast phase, scanner/kernel differences, dental materials, pediatric data, rare
  implants, and severe truncation require dedicated cohorts.
- NIfTI output is deliberate; safe DICOM-derived object creation and TPS integration
  require a separately verified clinical system.

## Primary sources

- [TCIA HNC-IMRT-70-33 collection](https://www.cancerimagingarchive.net/collection/hnc-imrt-70-33/)
- [TCIA HEAD-NECK-PET-CT collection](https://www.cancerimagingarchive.net/collection/head-neck-pet-ct/)
- [TCIA HNSCC collection](https://www.cancerimagingarchive.net/collection/hnscc/)
- [DCDiff, MICCAI 2024](https://papers.miccai.org/miccai-2024/192-Paper1608.html)
- [Dual-domain diffusion guidance for 3D CBCT MAR, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Choi_Dual_Domain_Diffusion_Guidance_for_3D_CBCT_Metal_Artifact_Reduction_WACV_2024_paper.html)
- [Masked-loss head-and-neck CT MAR, IEEE Access 2025](https://doi.org/10.1109/ACCESS.2025.3583191)
- [DASMamba medical image restoration, MICCAI 2025](https://papers.miccai.org/miccai-2025/paper/0433_paper.pdf)
- [Qwen3-Next architecture efficiency report](https://qwen.ai/blog?id=e34c4305036ce60d55a0791c170337c2b70ae51d)
- [Joint ESTRO/AAPM AI radiotherapy guideline](https://doi.org/10.1016/j.radonc.2024.110650)
