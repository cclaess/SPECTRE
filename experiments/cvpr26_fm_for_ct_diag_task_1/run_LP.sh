#!/bin/bash
set -e

# Default paths and settingsfor Docker environment
EMBEDS_ROOT="${EMBEDS_ROOT:-/workspace/outputs}"
LABELS_ROOT="${LABELS_ROOT:-/workspace/labels}"

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
    echo "Running linear probing for ${disease} ..."
    python3 run_LP.py \
        --embeds_root "$EMBEDS_ROOT" \
        --labels_root "$LABELS_ROOT" \
        --target "$disease" \
        --out_dir "$EMBEDS_ROOT/$disease/results" 
done