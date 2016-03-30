#!/bin/bash

#
# This code extracts CNN feature descriptors from a given dataset.
# Configuration variables are set below
#

datasets=(caltech10 office/amazon10 office/dslr10 office/webcam10)
DATA_DIR=$1
network_data_dir=$DATA_DIR/network/
PATCH_EXTRACTION_METHOD=extra #${patch_method[i]}
PSIZE=64
LEVEL=1
PATCHES_PER_IMAGE=64
IMAGE_DIM=227
DATA_SPLIT=-1 #-1 means all images, will ignore NUM_TEST and NUM_TRAIN
NUM_TRAIN=100 #100
NUM_TEST=100 #100
DECAF_LAYER_NAME=fc7_cudanet_out

for DATASET in "${datasets[@]}"; do
    INPUT_DIR=$DATA_DIR/images/$DATASET
    OUTPUT_DIR=$2/$DATASET/
    echo $OUTPUT_DIR
    for f in $INPUT_DIR/*; do
    echo $f
    python ./src/extract.py --input-dir $f --output-dir $OUTPUT_DIR \
        --patches $PATCHES_PER_IMAGE --patch-size $PSIZE \
        --image-dim $IMAGE_DIM --descriptor DECAF \
        --levels $LEVEL --split $DATA_SPLIT \
        --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST \
        --layer-name $DECAF_LAYER_NAME --network-data-dir $network_data_dir \
        --patch-method $PATCH_EXTRACTION_METHOD
    done
 done


