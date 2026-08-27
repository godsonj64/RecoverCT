# Data card

## Intended data

De-identified head-and-neck radiotherapy planning CT in DICOM, converted to floating-
point NIfTI while preserving HU and geometry. RTSTRUCT/SEG/RTPLAN/RTDOSE are retained
outside this repository for mask generation and downstream dosimetric validation.

## Manifest schema

`image_path`, `patient_id`, `series_uid`, `source_collection`, `split`, dimensions and
spacing, `metal_fraction`, `air_fraction`, `qc_pass`, `manual_approved`, and `qc_notes`.
The manifest is sensitive research metadata and is git-ignored.

For an institutionally curated paired real-artifact stage, provide `input_path`,
`target_path`, and `artifact_mask_path` together. All three NIfTI files must have the
same shape and affine after reviewed registration. `image_path` remains the clean target
field used by synthetic pretraining/fine-tuning rows.

## Quality labels

`qc_pass` is a coarse automatic screen. `manual_approved` means a qualified reviewer
has checked the actual volume and metadata against the study protocol. Neither label
means disease-free. Metal-free ground truth must not contain implant or streak artifact
in the supervised target region.

## Restrictions

Each TCIA collection has its own current access and citation requirements. Some facial
imaging is controlled to reduce re-identification risk. The downloader does not bypass
authentication or accept licenses for the user. Data must remain outside git and be
protected according to institutional policy.
