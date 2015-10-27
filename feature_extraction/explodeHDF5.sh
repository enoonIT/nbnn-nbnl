#!/bin/bash

#
# This code creates training and testing splits from previously extraced image features
#  --relu sets all negative features to 0

INPUT_DIR=$1
OUTPUT_DIR=$2
PATCHES_PER_IMAGE=100

python ./src/explode_hdf5.py --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR \
		--patches $PATCHES_PER_IMAGE
