#!/bin/bash

#
# This code extracts CNN feature descriptors from a given dataset.
# Configuration variables are set below
#
patch_size=(16 32 64)
levels=(1 2 3)
#patch_method=(base extra extra)
datasets=(caltech10 office/amazon office/dslr office/webcam)
COMBINATIONS=$((3*3)) #number of combinations for each dataset
DATA_DIR=$1
network_data_dir=$DATA_DIR/network/
i=$2
PATCH_EXTRACTION_METHOD=extra #${patch_method[i]}
DATASET=${datasets[$((i/COMBINATIONS))]}
PSIZE=${patch_size[$(((i % COMBINATIONS)/3))]}
LEVEL=${levels[$(((i % COMBINATIONS)%3))]}
INPUT_DIR=$DATA_DIR/images/$DATASET
OUT_NAME=all_${PSIZE}_${LEVEL}_${PATCH_EXTRACTION_METHOD}_hybrid_mean
OUTPUT_DIR=$DATA_DIR/desc/$DATASET/$OUT_NAME

echo $OUTPUT_DIR

PATCHES_PER_IMAGE=100
IMAGE_DIM=200
DATA_SPLIT=-1 #-1 means all images, will ignore NUM_TEST and NUM_TRAIN
NUM_TRAIN=100 #100
NUM_TEST=100 #100
DECAF_LAYER_NAME=fc7_cudanet_out

for f in $INPUT_DIR/*; do
  echo $f
python ./src/extract.py --input-dir $f --output-dir $OUTPUT_DIR \
       --patches $PATCHES_PER_IMAGE --patch-size $PSIZE \
       --image-dim $IMAGE_DIM --descriptor DECAF \
       --levels $LEVEL --split $DATA_SPLIT \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST \
       --layer-name $DECAF_LAYER_NAME --network-data-dir $network_data_dir \
       --patch-method $PATCH_EXTRACTION_PATCH_EXTRACTION_METHOD
done