FROM nvcr.io/nvidia/pytorch:25.02-py3

RUN apt-get update && apt-get install -y git

COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt

COPY spectre /app/
COPY experiments /app/

WORKDIR /app
ENV PYTHONPATH="${PYTHONPATH}:/app"