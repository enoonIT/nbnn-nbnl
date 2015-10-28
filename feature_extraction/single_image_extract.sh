#!/bin/bash

#
# This code extracts DECAF feature descriptors from patches of scene15 dataset.
# Configuration variables are set below (see extract_sports.sh for descriptions).
# Usage ./extract_scene15.sh /input/data_dir jobId /destination/dir


DATA_DIR=$1
network_data_dir=${2}/network/
PATCH_EXTRACTION_METHOD=extra #${patch_method[i]}
DATASET=scene15
PSIZE=32
LEVEL=3
INPUT_DIR=$DATA_DIR
OUT_NAME=single_image_${PSIZE}_${LEVEL}_${PATCH_EXTRACTION_METHOD}
OUT_PARENT=$DATA_DIR
if [ "$#" -gt 2 ]; then
  OUT_PARENT=$3
fi
OUTPUT_DIR=${OUT_PARENT}/$OUT_NAME

echo $OUTPUT_DIR

PATCHES_PER_IMAGE=100
IMAGE_DIM=200
DECAF_LAYER_NAME=fc7_cudanet_out


python ./src/single_image_extract.py --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR \
       --patches $PATCHES_PER_IMAGE --patch-size $PSIZE \
       --image-dim $IMAGE_DIM --descriptor DECAF \
       --levels $LEVEL \
       --layer-name $DECAF_LAYER_NAME --network-data-dir $network_data_dir \
       --patch-method $PATCH_EXTRACTION_METHOD