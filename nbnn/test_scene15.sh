#!/bin/bash

#
# This code tests NBNN classifier on a scenes15 dataset.
#

DATA_DIR=$1
NUM_TRAIN=100
NUM_TEST=100
ALPHA=100

START=1
END=5

for ((SPLIT = START; SPLIT <= END; SPLIT++))
  do
  echo "Split " $SPLIT
  python ./src/classify.py \
	--cmd classify \
	--train-dir $DATA_DIR/train/split_$SPLIT \
	--support $DATA_DIR/train/split_$SPLIT \
	--test-dir $DATA_DIR/test/split_$SPLIT \
	--num-train-images $NUM_TRAIN \
	--num-test-images $NUM_TEST \
	--alpha $ALPHA --patch_name "patches"
    done

