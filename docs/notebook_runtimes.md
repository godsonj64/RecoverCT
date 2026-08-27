# Notebook and cloud runtime guide

## Supported paths

The same clone supports local Jupyter, Google Colab, and RunPod Pods. The bootstrap
detects the platform and chooses a data root in this order:

1. `CT_RESTORE_DATA_ROOT` when explicitly set;
2. `/workspace/ct_restore_data` on RunPod;
3. `/content/ct_restore_data` on Colab;
4. `<repo>/data` locally.

RunPod volume or network-volume storage mounted at `/workspace` is appropriate for
datasets and checkpoints that must survive restarts. Container storage is temporary.
On Colab, `/content` is fast but ephemeral. Google Drive is persistent but can be a
data-loading bottleneck; a useful pattern is keeping checkpoints on Drive while copying
preprocessed training volumes to local storage for the active session.

## Installation behavior

`scripts/bootstrap_notebook.sh`:

- preserves the CUDA-compatible PyTorch already supplied by the runtime when its
  version satisfies the project requirement;
- installs the project, preprocessing libraries, TensorBoard, notebook kernel, and
  plotting dependency;
- places Torch and pip caches beneath the selected data root;
- prints disk and GPU information plus `ct-restore doctor` output.

If PyTorch is absent, pip resolves it from the configured package index. For a custom
CUDA image, install the correct PyTorch wheel from the official selector before running
the bootstrap.

## End-to-end smoke command

```bash
python scripts/run_notebook_pipeline.py \
  --data-root /workspace/ct_restore_data \
  --collection Pancreas-CT \
  --limit 1 \
  --epochs 1 \
  --accept-data-terms
```

The command performs:

1. TCIA NBIA metadata query and planning-series filtering;
2. resumable, CRC-checked ZIP download with a 10 GB free-space guard;
3. DICOM identifier/HU/geometry validation and resampling;
4. an automatically screened, clearly labeled smoke manifest;
5. one or more training epochs with hardware auto-tuning;
6. checkpointed full-volume inference and uncertainty output.

`--skip-download` resumes from an existing raw directory. Partial series ZIPs remain as
hidden `.zip.part` files and use HTTP range requests on the next run. Extracted series
are marked by `.complete`. An existing incomplete extracted directory is never silently
overwritten.

## Full experiment

The notebook smoke manifest is not a substitute for data curation. For a full run:

1. allocate at least 75 GB for the primary collection plus processed data/checkpoints;
2. query and download with `ct-restore tcia` (`--limit 0` means every selected series);
3. preprocess with `ct-restore preprocess`;
4. inspect every proposed clean target and set `manual_approved=true` only when valid;
5. train `configs/pretrain.yaml`, then `configs/finetune.yaml`;
6. use `configs/paired_real.yaml` only for reviewed registered real pairs;
7. perform the locked external and dosimetric validation protocol.

For multiple GPUs:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m ct_restore.cli train configs/finetune.yaml
```

## CUDA tuning and reproducibility

Auto-tuning changes patch dimensions, base channels, effective batch accumulation, and
workers according to detected VRAM/CPU capacity. It is enabled only in
`configs/notebook.yaml`; the main scientific configs keep `auto_tune: false` so an
experiment does not silently change between machines. Every run writes the resolved
configuration and hardware report to logs/checkpoints.

If an out-of-memory error occurs because other processes occupy VRAM, lower the patch
dimensions or base channels in a copied config. Do not reduce spatial resolution or HU
range without treating it as a new experiment. `torch.compile` is optional because
compile time, graph support, and speedup vary across Colab and RunPod GPU types.

## Data governance

Facial CT may be subject to controlled-access or noncommercial terms. The code does not
authenticate around restrictions or accept terms on the user's behalf. Keep patient
data out of git, use encrypted/policy-compliant storage, preserve source DICOM, and
follow the current collection page, institutional approval, and TCIA citation policy.

