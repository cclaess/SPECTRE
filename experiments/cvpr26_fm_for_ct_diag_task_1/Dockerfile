FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN groupadd -r user && useradd --no-log-init -m -r -g user user

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    git ffmpeg libsm6 libxext6 tzdata curl \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace/inputs /workspace/outputs \
    && chown user:user /workspace/inputs /workspace/outputs

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    pip-tools \
    monai \
    SimpleITK \
    tqdm \
    numpy \
    torchmetrics \
    pandas \
    matplotlib \
    wandb \
    spectre-fm

# Re-apply Torch pinning in case a transitive dependency attempted an upgrade.
RUN pip install --no-cache-dir --force-reinstall \
    --index-url https://download.pytorch.org/whl/cu124 \
    --extra-index-url https://pypi.org/simple \
    torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# Build-time sanity check: this should always stay on cu124 in this image.
RUN python -c "import sys, torch, numpy; print(sys.version); print(torch.__version__, torch.version.cuda, numpy.__version__); assert torch.version.cuda == '12.4'"

COPY --chown=user:user . /app
RUN mkdir -p /app/weights \
    && chown user:user /app/weights
RUN curl -L -o /app/weights/backbone_weights.pt \
    https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_backbone_vit_large_patch16_128.pt?download=true && \
    curl -L -o /app/weights/combiner_weights.pt \
    https://huggingface.co/cclaess/SPECTRE/resolve/main/spectre_combiner_feature_vit_large.pt?download=true && \
    chmod -R 755 /app/weights \
    && chown -R user:user /app

USER user

WORKDIR /app