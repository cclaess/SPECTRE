# 🌟 SPECTRE: Self-supervised Pretraining for CT Representation Extraction

SPECTRE is a framework for training and evaluating **transformer-based foundation models for 3D Computed Tomography (CT) scans**, and provdes a set of pretrained CT foundation model open to the research community. By leveraging self-supervised learning, SPECTRE enables the extraction of rich and generalizable representations from medical imaging data, which can be fine-tuned for various downstream tasks such as segmentation, classification, and anomaly detection.

SPECTRE has been trained on a large cohort of **open-source CT scans** of the **human abdomen and thorax**, as well as **paired scans with radiology reports**, enabling it to learn rich representations that generalize across different datasets and clinical settings.

This repository provides tools for pretraining, fine-tuning, and evaluating these models, ensuring robust performance across different CT datasets.

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
docker build -t spectre-ct .
```

### **Running Experiments Inside Docker**
Once the image is built, you can start a container and execute scripts inside it. For example, to run a DINO pretraining experiment:
```bash
docker run --gpus all --rm -v $(pwd):/app spectre-ct python3 experiments/pretraining/pretrain_dino.py --config_file spectre/configs/dino_default.yaml --output_dir outputs/pretraining/dino/
```
- `--gpus all` enables GPU acceleration if available.
- `--rm` removes the container after execution.
- `-v $(pwd):/app` mounts the current directory inside the container.
- `python3 experiments/pretraining/pretrain_dino.py --config_file spectre/configs/dino_default.yaml --output_dir outputs/pretraining/dino/` runs the DINO pretraining script with the default configuration file and stores the models weights in an output folder.


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
This project builds upon prior work in self-supervised learning, medical imaging, and transformer-based representation learning. We acknowledge the **open-source CT datasets** and **research, code, and packgaes** that made this research possible, including:
- [CT-RATE and CT-CLIP](https://arxiv.org/abs/2403.17834) – Thoracic CT scans with paired radiology reports, a CT foundation model based on the CLIP framework.
- [MERLIN](https://arxiv.org/abs/2406.06512) – Abdominal CT scans with paired radiology reports and ICD codes, a vision-language CT foundation model.
