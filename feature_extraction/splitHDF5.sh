#!/bin/bash

#
# This code creates training and testing splits from previously extraced image features
#  --relu sets all negative features to 0

INPUT_DIR=$1
OUTPUT_DIR=$1/splits
if [ "$#" -gt 1 ]; then
  OUTPUT_DIR=$1/splits_relu
fi
NUM_SPLITS=5
NUM_TRAIN=80	
NUM_TEST=20
PATCHES_PER_IMAGE=500

python ./src/makeSplits.py --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR \
       --num-splits $NUM_SPLITS --patches $PATCHES_PER_IMAGE \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST \
       $2
