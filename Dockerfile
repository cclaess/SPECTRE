FROM nvcr.io/nvidia/pytorch:25.03-py3

RUN apt-get update && apt-get install -y git

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY spectre /app/spectre
COPY experiments /app/experiments

WORKDIR /app
ENV PYTHONPATH="${PYTHONPATH}:/app"