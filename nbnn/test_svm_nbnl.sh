#!/bin/bash

#
# This code tests a version of NBNL which uses a linear svm
#

DATA_DIR=$1
NUM_TRAIN=$2
NUM_TEST=$3

START=1
END=1

for ((SPLIT = START; SPLIT <= END; SPLIT++))
  do
  echo "Split " $SPLIT
  python ./src/svm_baseline.py \
	--input-dir $DATA_DIR \
	--num-train-images $NUM_TRAIN \
	--num-test-images $NUM_TEST \
	--patch_name "patches7" --cmd "svm-nbnl" \
	--patches-per-image 350
    done

