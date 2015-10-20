#!/bin/bash

#
# This code tests NBNN classifier on a scenes15 dataset.
#

DATA_DIR=$1
NUM_TRAIN=50
NUM_TEST=50
ALPHA=100

python ./src/classify.py \
      --cmd classify \
      --test-dir $DATA_DIR \
      --num-train-images $NUM_TRAIN \
      --num-test-images $NUM_TEST \
      --alpha $ALPHA --patch_name "patches7" \
      --on_the_fly_splits


