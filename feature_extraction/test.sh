#!/bin/bash
DATA_DIR=$1
i=$2
DATASET=scene15
INPUT_DIR=$DATA_DIR/images/$DATASET

PATCHES_PER_IMAGE=100
PATCH_SIZE=32
IMAGE_DIM=200
LEVELS=3
DATA_SPLIT=-1
NUM_TRAIN=100
NUM_TEST=100
DECAF_LAYER_NAME=fc7_cudanet_out

patch_size=(32 32 16)
levels=(3 3 4)
patch_method=(base extra extra)
items=${#patch_size[@]}

OUT_NAME=all_${patch_size[i]}_${levels[i]}_${patch_method[i]}_hybrid_mean;
OUTPUT_DIR=$DATA_DIR/desc/$DATASET/$OUT_NAME;
echo $OUTPUT_DIR;

# for (( i=0; i < items; i++ )); #all_32_3_improved_hybrid_mean
# do 
#   OUT_NAME=all_${patch_size[i]}_${levels[i]}_${patch_method[i]}_hybrid_mean;
#   OUTPUT_DIR=$DATA_DIR/desc/$DATASET/$OUT_NAME;
#   echo $OUTPUT_DIR;
# done