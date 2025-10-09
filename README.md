# 🌟 SPECTRE: cross-modal Self-supervised Pretraining for CT Representation Extraction  

SPECTRE is a **transformer-based foundation model for 3D Computed Tomography (CT) scans**, trained using **self-supervised learning** and **cross-modal vision–language alignment**. It provides rich and generalizable representations from medical imaging data, which can be fine-tuned for downstream tasks such as segmentation, classification, and anomaly detection.  

SPECTRE has been trained on a large cohort of **open-source CT scans** of the **human abdomen and thorax**, as well as **paired radiology reports**, enabling it to capture representations that generalize across datasets and clinical settings.  

This repository provides pretrained SPECTRE models together with tools for fine-tuning and evaluation, ensuring robust performance across diverse CT datasets.

## 🧠 Pretrained Models

| Model Name                | Modality           | Pretraining Objective   | Model Weights     |
|---------------------------|--------------------|-------------------------|-------------------|
| **SPECTRE**               | **CT**             | **DINOv3 + SigLIP**     | **Link**          |
| SPECTRE-ViT-Local         | CT (crops)         | DINOv3                  | Link              |
| SPECTRE-ViT-Local         | CT (crops)         | DINOv3 + SigLIP         | Link              |
| SPECTRE-ViT-Global        | CT                 | DINOv3 + SigLIP         | Link              |
| Qwen3-Embedding-0.6B LoRA | Text (radiology)   | SigLIP                  | Link              |

## 📂 Repository Contents

This repository is organized as follows:

- 🚀 **`spectre/`** – Contains the core package, including:
  - Pretraining methods
  - Model architectures
  - Data transformations

- 🛠️ **`spectre/configs/`** – Stores configuration files for different training settings.

- 🔬 **`experiments/`** – Includes Python scripts for running various pretraining and downstream experiments.

- 📄 **`requirements.txt`** – Lists the dependencies required to run the project.

- 🐳 **`Dockerfile`** – Defines the environment for running SPECTRE inside a container.

## ⚙️ Setting Up the Environment

To get up and running with SPECTRE, follow these steps:

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/cclaess/SPECTRE.git
   cd SPECTRE
   ```

2. Create a `.env` file in the root directory of the project.

3. Add your [Weights & Biases](https://wandb.ai/) API key to the `.env` file:
   ```bash
   WANDB_API_KEY=<your_api_key>
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## ⚡ Performing Pretraining
...

## 🧩 Performing Downstream Training
...

## 🧪 Testing Your Algorithms
...

## 🐳 Building and Using Docker

To facilitate deployment and reproducibility, SPECTRE can be run using **Docker**. This allows you to set up a fully functional environment without manually installing dependencies.

### **Building the Docker Image**
First, ensure you have **Docker** installed. Then, navigate to the repository and build the image:
```bash
docker build -t spectre .
```

### **Running Experiments Inside Docker**
Once the image is built, you can start a container and execute scripts inside it. For example, to run a DINO pretraining experiment:
```bash
docker run --gpus all --rm -v $(pwd):/app spectre python3 experiments/pretraining/pretrain_dino.py --config_file spectre/configs/dino_default.yaml --output_dir $(pwd)/outputs/pretraining/dino/
```
- `--gpus all` enables GPU acceleration if available.
- `--rm` removes the container after execution.
- `-v $(pwd):/app` mounts the current directory inside the container.
- `python3 experiments/pretraining/pretrain_dino.py --config_file spectre/configs/dino_default.yaml --output_dir $(pwd)/outputs/pretraining/dino/` runs the DINO pretraining script with the default configuration file and stores the models weights in an output folder.


## 📜 Citation
If you use SPECTRE in your research or wish to cite it, please use the following BibTeX entry:
```
@article{spectre2025,
  title={Citation will be added upon publication},
  author={},
  journal={},
  year={2025},
  url={https://github.com/cclaess/SPECTRE}
}
```

## 🤝 Acknowledgements
This project builds upon prior work in self-supervised learning, medical imaging, and transformer-based representation learning. We acknowledge the **open-source CT datasets** and **research, code, and packages** that made this research possible, including:
- [**DINO**](https://arxiv.org/abs/2104.14294), [**DINOv2**](https://arxiv.org/abs/2304.07193), [**MAE**](https://arxiv.org/abs/2111.06377), & [**SigLIP**](https://arxiv.org/abs/2303.15343): Self-supervised vision and vision-language representation learning approaches that inspired this work.
- [**timm**](https://timm.fast.ai/) & [**lightly**](https://docs.lightly.ai/self-supervised-learning/): Python libraries providing 2D PyTorch models (timm) and self-supervised learning methods (lightly), from which we adapted parts of the code for 3D.
- [**CT-RATE & CT-CLIP**](https://arxiv.org/abs/2403.17834): A dataset of thoracic CT scans paired with radiology reports (CT-RATE) and a CT foundation model based on the CLIP framework trained on this dataset (CT-CLIP).
- [**MERLIN**](https://arxiv.org/abs/2406.06512): A dataset of abdominal CT scans paired with radiology reports and ICD10 codes, along with a vision-language CT foundation model trained on this dataset.
