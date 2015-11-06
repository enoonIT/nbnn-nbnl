#!/bin/bash

#
# This code creates training and testing splits from previously extraced image features
#  --relu sets all negative features to 0

SOURCE_DIR=$1
TARGET_DIR=$2
OUTPUT_DIR=$3
NUM_TRAIN=20	
NUM_TRANSF=$4
SPLIT=$5
                       
python ./src/da_splits.py --source-dir $SOURCE_DIR --target-dir $TARGET_DIR \
        --output-dir $OUTPUT_DIR --split $SPLIT \
       --train-images $NUM_TRAIN --transfer-imgs $NUM_TRANSF
