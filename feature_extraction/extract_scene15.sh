#!/bin/bash

#
# This code extracts DECAF feature descriptors from patches of scene15 dataset.
# Configuration variables are set below (see extract_sports.sh for descriptions).
# Note that descriptors are extracted only from one data split.
#
patch_size=(32 32 16)
levels=(3 3 4)
patch_method=(base extra extra)

DATA_DIR=$1
network_data_dir=$DATA_DIR/network/
i=$2
DATASET=test
INPUT_DIR=$DATA_DIR/images/$DATASET
OUT_NAME=all_${patch_size[i]}_${levels[i]}_${patch_method[i]}_hybrid_mean;
OUTPUT_DIR=$DATA_DIR/desc/$DATASET/$OUT_NAME;

PATCHES_PER_IMAGE=100
PATCH_SIZE=${patch_size[i]}
PATCH_EXTRACTION_METHOD=${patch_method[i]}
IMAGE_DIM=200
LEVELS=${levels[i]}
DATA_SPLIT=-1 #-1 means all images, will ignore NUM_TEST and NUM_TRAIN
NUM_TRAIN=100 #100
NUM_TEST=100 #100
DECAF_LAYER_NAME=fc7_cudanet_out

for f in $INPUT_DIR/*; do
  echo $f
python ./src/extract.py --input-dir $f --output-dir $OUTPUT_DIR \
       --patches $PATCHES_PER_IMAGE --patch-size $PATCH_SIZE \
       --image-dim $IMAGE_DIM --descriptor DECAF \
       --levels $LEVELS --split $DATA_SPLIT \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST \
       --layer-name $DECAF_LAYER_NAME --network-data-dir $network_data_dir \
       --patch-method $PATCH_EXTRACTION_METHOD
done

