#!/bin/bash

#
# This code tests NBNN classifier on a scenes15 dataset.
#

DATA_DIR=~/data/desc/caltech10/
NUM_TRAIN=20
NUM_TEST=0
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
	--alpha $ALPHA
    done

