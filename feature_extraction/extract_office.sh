#!/bin/bash

#
# This code extracts CNN feature descriptors from a given dataset.
# Configuration variables are set below
#
DATA_DIR=$1
network_data_dir=$DATA_DIR/network/
i=$2
PATCH_EXTRACTION_METHOD=extra #${patch_method[i]}
DATASET=office/amazon
PSIZE=32
LEVEL=3
INPUT_DIR=$DATA_DIR/images/$DATASET
OUT_NAME=all_${PSIZE}_${LEVEL}_${PATCH_EXTRACTION_METHOD}_hybrid_mean
OUTPUT_DIR=$DATA_DIR/desc/$DATASET/$OUT_NAME

echo $OUTPUT_DIR

PATCHES_PER_IMAGE=100
IMAGE_DIM=200
DATA_SPLIT=-1 #-1 means all images, will ignore NUM_TEST and NUM_TRAIN
NUM_TRAIN=100 #100
NUM_TEST=100 #100
DECAF_LAYER_NAME=67

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
