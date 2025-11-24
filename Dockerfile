FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy the application code
COPY . /app

# Install the application in editable mode
RUN pip install --no-cache-dir -e /app

# Set the working directory
WORKDIR /app
ENV PYTHONPATH="${PYTHONPATH}:/app"