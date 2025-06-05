#!/bin/bash

# Login to Weights & Biases
if [ -n "$WANDB_API_KEY" ]; then
    echo "Logging in to Weights & Biases..."
    wandb login --relogin $WANDB_API_KEY
else
    echo "No WANDB_API_KEY provided. Skipping Weights & Biases login."
fi

# Ensure that required system parameters are set, or exit with an error.
: "${MASTER_ADDR:?MASTER_ADDR is required}"
: "${MASTER_PORT:?MASTER_PORT is required}"
: "${WORLD_SIZE:?WORLD_SIZE is required}"
: "${RANK:?RANK is required}"

# Set defaults if not provided
CONFIG_FILE="${CONFIG_FILE:-spectre/configs/accelerate_default.yaml}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_MACHINES="${NUM_MACHINES:-1}"

echo "Using configuration:"
echo "  CONFIG_FILE:      $CONFIG_FILE"
echo "  NUM_PROCESSES:    $NUM_PROCESSES"
echo "  NUM_MACHINES:     $NUM_MACHINES"
echo "  MASTER_ADDR:   $MASTER_ADDR"
echo "  MASTER_PORT:   $MASTER_PORT"
echo "  WORLD_SIZE:    $WORLD_SIZE"
echo "  RANK:          $RANK"
echo "  IP:            $MASTER_IP"

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <script.py> [args...]"
    exit 1
fi

cuda-gdb -ex run --args python -u -m accelerate.commands.launch \
  --config_file $CONFIG_FILE \
  --num_processes $NUM_PROCESSES \
  --machine_rank $RANK \
  --num_machines $NUM_MACHINES \
  --main_process_port $MASTER_PORT \
  --main_process_ip $MASTER_ADDR \
  "$@"