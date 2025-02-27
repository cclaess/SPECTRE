# SPECTRE: Self-supervised Pretraining for CT Representation Extraction
SPECTRE is a framework for training and evaluating **transformer-based foundation models for 3D CT scans**. By leveraging self-supervised learning, SPECTRE enables the extraction of rich and generalizable representations from medical imaging data, which can be fine-tuned for various downstream tasks such as segmentation, classification, and anomaly detection.

This repository provides tools for pretraining, fine-tuning, and evaluating these models, ensuring robust performance across different CT datasets.

## Getting ready
To begin using SPECTRE, follow these steps:
1. Clone this repository to your machine:   
    ```bash
    git clone https://github.com/cclaess/SPECTRE.git
    cd SPECTRE
    ```
2. Create a `.env` file in the root directory of the project.
3. Add the following to the `.env` file, replacing `<your_api_key>` with your actual [Weights & Biases](https://wandb.ai/) API key:
    ```bash
    WANDB_API_KEY=<your_api_key>
    ```
4. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
...