📢 [2026-05-20] The pretrained SPECTRE model can now be loaded directly through the `transformers` library, no separate SPECTRE package installation required. Check below for details and usage examples.

📢 [2026-04-10] SPECTRE is now an official baseline for the [**CVPR 2026 Workshop Competition: Foundation Models for General CT Image Diagnosis**](https://www.codabench.org/competitions/12650/)! See `experiments/cvpr26_fm_for_ct_diag_task_1` for scripts and additional details.  

📢 [2026-02-21] SPECTRE has been accepted for presentation at **CVPR 2026** (Denver, Colorado, USA)!  

📢 [2026-01-20] [Semantic segmentation](https://github.com/cviviers/nnUNet) code and configurations using the nnUNet framework are now released!  


# SPECTRE 👻👻👻

<p align="center">
  <a href="https://pypi.org/project/spectre-fm/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/spectre-fm?style=flat-square&label=version&cacheSeconds=0" /></a>
  <a href="https://pypi.org/project/spectre-fm/"><img alt="Python Versions" src="https://img.shields.io/pypi/pyversions/spectre-fm?style=flat-square&cacheSeconds=0" /></a>
  <a href="https://pypi.org/project/spectre-fm/"><img alt="Downloads per Month" src="https://img.shields.io/pypi/dm/spectre-fm?style=flat-square&label=downloads&cacheSeconds=0" /></a>
  <a href="https://github.com/cclaess/SPECTRE/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/cclaess/SPECTRE?style=flat-square&cacheSeconds=0" /></a>
  <a href="https://huggingface.co/cclaess/SPECTRE-Large"><img alt="Model weights" src="https://img.shields.io/badge/model-Hugging%20Face-yellow?style=flat-square&cacheSeconds=0" /></a>
  <a href="https://arxiv.org/abs/2511.17209"><img alt="Preprint" src="https://img.shields.io/badge/preprint-arXiv-b31b1b?style=flat-square&cacheSeconds=0" /></a>
</p>

<p align="center">
   <img src="imgs/method_overview.jpg" alt="SPECTRE architecture and pretraining strategies" width="600"/>
</p>

SPECTRE (**S**elf-Supervised & Cross-Modal **P**r**e**training for **CT** **R**epresentation **E**xtraction) is a **Transformer-based foundation model for 3D Computed Tomography (CT) scans**, trained using **self-supervised learning** (SSL) and **cross-modal vision–language alignment** (VLA). It provides rich and generalizable representations from medical imaging data, which can be fine-tuned for downstream tasks such as segmentation, classification, and anomaly detection.  

SPECTRE has been trained on a large cohort of **open-source CT scans** of the **human abdomen and thorax**, as well as **paired radiology reports** and **Electronic Health Record data**, enabling it to capture representations that generalize across datasets and clinical settings.  

This repository provides pretrained SPECTRE models together with tools for fine-tuning and evaluation.

## 🧠 Pretrained Models
The pretrained SPECTRE model can easily be imported using the `transformers` library

```python
from transformers import AutoModel
model = AutoModel.from_pretrained('cclaess/SPECTRE-Large', trust_remote_code=True)
```

or by using the `spectre-fm` package as follows:

```python
from spectre import SpectreImageFeatureExtractor, MODEL_CONFIGS
config = MODEL_CONFIGS['spectre-large-pretrained']
model = SpectreImageFeatureExtractor.from_config(config)
```

A simple forward pass would look like:
```python
import torch

model.eval()

# Dummy input: (batch, crops, channels, height, width, depth)
# For a (3 x 3 x 4) grid of (128 x 128 x 64) CT patches -> Total scan size (384 x 384 x 256)
x = torch.randn(1, 1, 384, 384, 256)
B, C, H, W, D = x.shape

patch_size = (128, 128, 64)
pH, pW, pD = patch_size

x = x.view(
  B, C,
  H // pH, pH,
  W // pW, pW,
  D // pD, pD,
).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(B, -1, C, pH, pW, pD)

with torch.no_grad():
    features = model(
      x, 
      grid_size=(
        H // pH,
        W // pW,
        D // pD,
      ),
    )
print("Features shape:", features.shape)
```

Alternatively, you can download the weights of the separate components through HuggingFace using the following links:

| Architecture              | Input Modality     | Pretraining Objective   | Model Weights                                                                                                               |
|---------------------------|--------------------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| SPECTRE-ViT-Local         | CT crops           | SSL                     | [Link](https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_backbone_vit_large_patch16_128_no_vla.pt?download=true)  |
| SPECTRE-ViT-Local         | CT crops           | SSL + VLA               | [Link](https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_backbone_vit_large_patch16_128.pt?download=true)         |
| SPECTRE-ViT-Global        | Embedded CT crops  | VLA                     | [Link](https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_combiner_feature_vit_large.pt?download=true)             |
| Qwen3-Embedding-0.6B LoRA | Text (radiology)   | VLA                     | [Link](https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_qwen3_embedding_0.6B_lora.pt?download=true)              |

## 🩻 Segmentation (nnUNet)

If you're looking for a nnUNet-based segmentation pipeline that uses SPECTRE as the backbone, see: https://github.com/cviviers/nnUNet

## 📂 Repository Contents

This repository is organized as follows:

- 🚀 **`src/spectre/`** – Contains the core package, including:
  - Pretraining methods
  - Model architectures
  - Data handling and transformations

- 🛠️ **`src/spectre/configs/`** – Stores configuration files for different training settings.

- 🔬 **`experiments/`** – Includes Python scripts for running various pretraining and downstream experiments.

- 🐳 **`Dockerfile`** – Defines the environment for running a local version of SPECTRE inside a container.

## ⚙️ Setting Up the Environment

To get up and running with SPECTRE, install the base package with pip:

```bash
pip install spectre-fm
```

This installs only the runtime dependencies needed to load and run the pretrained models.

If you want to fine-tune or pretrain SPECTRE, install the matching extra:

```bash
pip install "spectre-fm[training]"
```

If you only need the evaluation stack, install:

```bash
pip install "spectre-fm[eval]"
```

If training on GDS-enabled systems is required, install the CUDA 12 specific extra:

```bash
pip install "spectre-fm[gds-cuda12]"  # with training stack: "spectre-fm[training,gds-cuda12]"
```

**Note that** `gds-cuda12` is only compatible with CUDA 12.x environments.

To install everything at once, use:

```bash
pip install "spectre-fm[all]"
```

or install the latest updates directly from GitHub:

```bash
pip install git+https://github.com/cclaess/SPECTRE.git
```

## 🐳 Building and Using Docker

To facilitate deployment and reproducibility, SPECTRE can be run using **Docker**. This allows you to set up a fully functional environment without manually installing dependencies using your own local copy of spectre.

### **Building the Docker Image**
First, ensure you have **Docker** installed. Then, clone and navigate to the repository to build the image:
```bash
git clone https://github.com/cclaess/SPECTRE
cd SPECTRE
docker build -t spectre-fm .
```

### **Running Experiments Inside Docker**
Once the image is built, you can start a container and execute scripts inside it. For example, to run a DINO pretraining experiment:
```bash
docker run --gpus all --rm -v "$(pwd):/mnt" spectre-fm python3 experiments/pretraining/pretrain_dino.py --config_file spectre/configs/dino_default.yaml --output_dir /mnt/outputs/pretraining/dino/
```
- `--gpus all` enables GPU acceleration if available.
- `--rm` removes the container after execution.
- `-v $(pwd):/mnt` mounts the current directory inside the container.

## ⚖️ License
- **Code: MIT** — see `LICENSE` (permissive; commercial use permitted).
- **Pretrained model weights: CC-BY-NC-SA** — non-commercial share-alike. The weights and any derivative models that include these weights are NOT cleared for commercial use. See `LICENSE_MODELS` for details and the precise license text.

> Note: the pretrained weights are subject to the original dataset licenses. Users intending to use SPECTRE in commercial settings should verify dataset and model licensing and obtain any required permissions.

## 📜 Citation
If you use SPECTRE in your research or wish to cite it, please use the following BibTeX entry of our [preprint](https://arxiv.org/abs/2511.17209):
```
@misc{claessens_scaling_2025,
  title = {Scaling {Self}-{Supervised} and {Cross}-{Modal} {Pretraining} for {Volumetric} {CT} {Transformers}},
  url = {http://arxiv.org/abs/2511.17209},
  doi = {10.48550/arXiv.2511.17209},
  author = {Claessens, Cris and Viviers, Christiaan and D'Amicantonio, Giacomo and Bondarev, Egor and Sommen, Fons van der},
  year={2025},
}
```

## 🤝 Acknowledgements
This project builds upon prior work in self-supervised learning, medical imaging, and transformer-based representation learning. We especially acknowledge [**MONAI**](https://project-monai.github.io/) for their awesome framework and the [**timm**](https://timm.fast.ai/) & [**lightly**](https://docs.lightly.ai/self-supervised-learning/) Python libraries for providing 2D PyTorch models (timm) and object-oriented self-supervised learning methods (lightly), from which we adapted parts of the code for 3D.

[![Star History Chart](https://api.star-history.com/svg?repos=cclaess/SPECTRE&type=Date)](https://star-history.com/#cclaess/SPECTRE&Date)