FROM nvcr.io/nvidia/pytorch:25.04-py3

# Set non-interactive mode for apt
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y git

# Upgrade pip
RUN pip install --upgrade pip

# Copy and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the application code
COPY spectre /app/spectre
COPY experiments /app/experiments
COPY scripts /app/scripts
COPY tests /app/tests

# Set the working directory
WORKDIR /app
ENV PYTHONPATH="${PYTHONPATH}:/app"