#!/bin/bash

#
# This code creates training and testing splits from previously extraced image features
#  --relu sets all negative features to 0

INPUT_DIR=$1
OUTPUT_DIR=data/desc/sports
NUM_SPLITS=5
NUM_TRAIN=100	
NUM_TEST=100
PATCHES_PER_IMAGE=100

python ./src/makeSplits.py --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR \
       --num-splits $NUM_SPLITS --patches $PATCHES_PER_IMAGE \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST --relu