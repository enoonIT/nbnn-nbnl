#!/bin/bash

#
# This code extracts DECAF feature descriptors from patches of scene15 dataset.
# Configuration variables are set below (see extract_sports.sh for descriptions).
# Note that descriptors are extracted only from one data split.
#

INPUT_DIR=data/images/scene15
OUTPUT_DIR=data/desc/scene15/all_32_3_improved_hybrid_mean
PATCHES_PER_IMAGE=100
PATCH_SIZE=32
IMAGE_DIM=200
LEVELS=3
DATA_SPLIT=-1
NUM_TRAIN=100
NUM_TEST=100
DECAF_LAYER_NAME=fc7_cudanet_out

for f in $INPUT_DIR/*; do
#   if [ "$f" != "data/images/scene15/industrial" ]; then
#     continue
#   fi
  echo $f
python ./src/extract.py --input-dir $f --output-dir $OUTPUT_DIR \
       --patches $PATCHES_PER_IMAGE --patch-size $PATCH_SIZE \
       --image-dim $IMAGE_DIM --descriptor DECAF \
       --levels $LEVELS --split $DATA_SPLIT \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST \
       --layer-name $DECAF_LAYER_NAME
done

