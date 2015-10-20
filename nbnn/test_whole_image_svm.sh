#!/bin/bash

#
# This code tests NBNN classifier on a scenes15 dataset.
#

DATA_DIR=$1
NUM_TRAIN=$2
NUM_TEST=$3

START=1
END=5

for ((SPLIT = START; SPLIT <= END; SPLIT++))
  do
  echo "Split " $SPLIT
  python ./src/svm_baseline.py \
	--input-dir $DATA_DIR \
	--num-train-images $NUM_TRAIN \
	--num-test-images $NUM_TEST \
	--patch_name "patches7"
    done

