#!/bin/bash

#
# This code extracts CNN feature descriptors from a given dataset.
# Configuration variables are set below
#
DATA_DIR=$1
network_data_dir=$DATA_DIR/network/
PATCH_EXTRACTION_METHOD=extra #${patch_method[i]}
DATASET=street_train
PSIZE=64
LEVEL=1

INPUT_DIR=$2
OUT_NAME=patch_grid_${PSIZE}
OUTPUT_DIR=$DATA_DIR/desc/$DATASET/$OUT_NAME

echo $OUTPUT_DIR

PATCHES_PER_IMAGE=64
IMAGE_DIM=256
DATA_SPLIT=-1 #-1 means all images, will ignore NUM_TEST and NUM_TRAIN
NUM_TRAIN=100 #100
NUM_TEST=100 #100
DECAF_LAYER_NAME=fc7_cudanet_out

python ./src/single_image_extract.py --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR \
    --patches $PATCHES_PER_IMAGE --patch-size $PSIZE \
    --image-dim $IMAGE_DIM --descriptor DECAF \
    --levels $LEVEL \
    --layer-name $DECAF_LAYER_NAME --network-data-dir $network_data_dir \
    --patch-method $PATCH_EXTRACTION_METHOD

