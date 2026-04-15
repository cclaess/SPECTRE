# SPECTRE Baseline for CVPR 2026 CTFM Competition (Task 1: Linear Probing)

This folder contains the scripts and Docker setup used to run the SPECTRE baseline submitted to the CVPR 2026 Workshop competition:

- Foundation Models for General CT Image Diagnosis (Task 1: Linear Probing)
- Challenge page: https://www.codabench.org/competitions/12650/

The same workflow can be reused to reproduce baseline results and run your own feature extraction, linear-probe training, and validation inference.

## Contents

- `Dockerfile`: Docker image definition used for the challenge baseline environment.
- `extract_feat_LP.py` / `extract_feat_LP.sh`: Feature extraction pipeline.
- `run_LP.py` / `run_LP.sh`: Linear probing (classifier training).
- `cvpr26_inference_LP.py` / `cvpr26_inference_LP.sh`: Validation-set inference.
- `datasets.py`, `metrics.py`: Dataset and metric utilities.

## Prerequisites

- Docker with NVIDIA GPU support (NVIDIA Container Toolkit).
- Access to competition datasets and labels on your local machine.
- Sufficient disk space for extracted features and training outputs.

## 1) Build the Docker image

From this folder, build the image with:

```bash
docker build -t spectre_lp:latest .
```

You can either run this locally built image (`spectre_lp:latest`) or the uploaded baseline image (`cclaess/cvpr_ctfm_comp:v1`).

## 2) Extract features

Example for LUNA25 feature extraction:

```bash
docker container run --gpus "device=0" --rm \
	-v ~/Datasets/CVPR2026-3DCTFMCompetition/LUNA25/images/:/workspace/inputs/ \
	-v ~/Datasets/CVPR-3DCTFMCompetition/LUNA25/features/:/workspace/outputs/ \
	spectre_lp:latest /bin/bash -c "sh extract_feat_LP.sh"
```

If you want to use the already built image from DockerHub, replace `spectre_lp:latest` with `cclaess/cvpr_ctfm_comp:v1` after pulling the image.

## 3) Train the linear classifier (linear probing)

Example command (task: `lung_nodule_malignancy`):

```bash
docker container run --gpus "device=0" --rm \
	-v ~/Datasets/CVPR2026-3DCTFMCompetition/LUNA25/features/:/workspace/outputs/ \
	-v ~/Datasets/CVPR2026-3DCTFMCompetition/LUNA25/labels/:/workspace/labels/ \
	spectre_lp:latest /bin/bash -c "bash run_LP.sh lung_nodule_malignancy"
```

## 4) Run validation inference

Example command:

```bash
docker container run --gpus "device=0" --rm \
	-v ~/Datasets/CVPR2026-3DCTFMCompetition/LUNA25/features/:/workspace/outputs/ \
	-v ~/Datasets/CVPR2026-3DCTFMCompetition/LUNA25/labels/:/workspace/labels/ \
	spectre_lp:latest /bin/bash -c "bash cvpr26_inference_LP.sh lung_nodule_malignancy"
```

## Notes

- Ensure your mounted folders contain the expected files before long runs.
- Replace the paths to the datasets with your local paths.
- Keep mount targets consistent with the scripts:
	- Inputs go to `/workspace/inputs/`
	- Extracted features go to `/workspace/outputs/`
	- Labels go to `/workspace/labels/`
	- Training outputs go to `/workspace/results/`
- For PowerShell users on Windows, replace `~` with `$HOME` or, when mounting behaves unexpectedly, use full absolute paths for `-v` mounts.

## Citation and Attribution

If you use this baseline in your work, please cite the SPECTRE repository and reference the CVPR 2026 CTFM competition page.
