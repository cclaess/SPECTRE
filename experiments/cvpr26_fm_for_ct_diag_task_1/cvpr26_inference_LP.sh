#!/bin/bash
set -e

# Default paths and settings for Docker environment
EMBEDS_ROOT="${EMBEDS_ROOT:-/workspace/outputs}"
LABELS_ROOT="${LABELS_ROOT:-/workspace/labels}"
SPLIT="${SPLIT:-val}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-16}"


# If arguments for diseases are provided, use them; otherwise, use the default list
if [ "$#" -gt 0 ]; then
    disease_list=("$@")
else
  disease_list=(
    splenomegaly
    adrenal_hyperplasia
    fatty_liver
    cholecystitis
    liver_calcifications
    hydronephrosis
    gallstone
    liver_lesion
    kidney_stone
    liver_cyst
    renal_cyst
    atherosclerosis
    colorectal_cancer
    ascites
    lymphadenopathy
  )
fi

for disease in "${disease_list[@]}"; do
    echo "Running inference for ${disease} on ${SPLIT} split ..."
    python3 cvpr26_inference_LP.py \
        --embeds_root "$EMBEDS_ROOT" \
        --labels_root "$LABELS_ROOT" \
        --target "$disease" \
        --split "$SPLIT" \
        --ckpt_dir "$EMBEDS_ROOT/$disease/results" \
        --batch_size $BATCH_SIZE \
        --num_workers $NUM_WORKERS
    echo ""
done