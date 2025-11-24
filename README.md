# SPECTRE - Self-Supervised & Cross-Modal Pretraining for CT Representation Extraction  

<p align="center">
   <img src="imgs/method_overview.jpg" alt="SPECTRE architecture and pretraining strategies" width="600"/>
</p>

SPECTRE is a **Transformer-based foundation model for 3D Computed Tomography (CT) scans**, trained using **self-supervised learning** (SSL) and **cross-modal vision–language alignment** (VLA). It provides rich and generalizable representations from medical imaging data, which can be fine-tuned for downstream tasks such as segmentation, classification, and anomaly detection.  

SPECTRE has been trained on a large cohort of **open-source CT scans** of the **human abdomen and thorax**, as well as **paired radiology reports** and **Electronic Health Record data**, enabling it to capture representations that generalize across datasets and clinical settings.  

This repository provides pretrained SPECTRE models together with tools for fine-tuning and evaluation.

## 🧠 Pretrained Models
The pretrained SPECTRE model can easily be imported as follows:

```python
from spectre import SpectreImageFeatureExtractor, MODEL_CONFIGS
import torch

config = MODEL_CONFIGS['spectre-large-pretrained']
model = SpectreImageFeatureExtractor.from_config(config)
model.eval()

# Dummy input: (batch, crops, channels, height, width, depth)
# For a (3 x 3 x 4) grid of (128 x 128 x 64) CT patches -> Total scan size (384 x 384 x 256)
x = torch.randn(1, 36, 1, 128, 128, 64)
with torch.no_grad():
    features = model(x, grid_size=(3, 3, 4))
print("Features shape:", features.shape)
```

Alternatively, you can download the weights of the separate components through HuggingFace using the following links:

| Architecture              | Input Modality     | Pretraining Objective   | Model Weights                                                                                                               |
|---------------------------|--------------------|-------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| SPECTRE-ViT-Local         | CT crops           | SSL                     | [Link](https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_backbone_vit_large_patch16_128_no_vla.pt?download=true)  |
| SPECTRE-ViT-Local         | CT crops           | SSL + VLA               | [Link](https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_backbone_vit_large_patch16_128.pt?download=true)         |
| SPECTRE-ViT-Global        | Embedded CT crops  | VLA                     | [Link](https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_combiner_feature_vit_large.pt?download=true)             |
| Qwen3-Embedding-0.6B LoRA | Text (radiology)   | VLA                     | [Link](https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_qwen3_embedding_0.6B_lora.pt?download=true)              |

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

To get up and running with SPECTRE, simply install our package using pip:

```bash
pip install spectre
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
docker build -t spectre .
```

### **Running Experiments Inside Docker**
Once the image is built, you can start a container and execute scripts inside it. For example, to run a DINO pretraining experiment:
```bash
docker run --gpus all --rm -v "$(pwd):/mnt" spectre python3 experiments/pretraining/pretrain_dino.py --config_file spectre/configs/dino_default.yaml --output_dir /mnt/outputs/pretraining/dino/
```
- `--gpus all` enables GPU acceleration if available.
- `--rm` removes the container after execution.
- `-v $(pwd):/mnt` mounts the current directory inside the container.

## License
- Code: MIT License — see `LICENSE` (permissive; commercial use permitted).
- Pretrained model weights: CC-BY-NC-SA — non-commercial share-alike. The weights and any derivative models that include these weights are NOT cleared for commercial use. See `LICENSE_MODELS.md` for details and the precise license text.

> Note: the pretrained weights are subject to the original dataset licenses. Users intending to use SPECTRE in clinical or commercial settings should verify dataset and model licensing and obtain any required permissions.

## 📜 Citation
If you use SPECTRE in your research or wish to cite it, please use the following BibTeX entry of our preprint:
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
This project builds upon prior work in self-supervised learning, medical imaging, and transformer-based representation learning. We especially acknowledge the [**timm**](https://timm.fast.ai/) & [**lightly**](https://docs.lightly.ai/self-supervised-learning/) Python libraries for providing 2D PyTorch models (timm) and object-oriented self-supervised learning methods (lightly), from which we adapted parts of the code for 3D.

<p align="center">
   <img src="imgs/cover_image.jpg" alt="SPECTRE cover image" width="600"/>
</p>