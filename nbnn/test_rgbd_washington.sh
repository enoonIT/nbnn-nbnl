#!/bin/bash

#
# This code tests a linear svm on the whole images, using a CNN based feature
#

DATA_DIR="/home/fmcarlucci/data/desc/rgbd-dataset_eval_depth/" 
SPLIT_FILE="/home/fmcarlucci/data/desc/rgbd-dataset_eval/testinstance_ids.txt"


START=1
END=10

for ((SPLIT = START; SPLIT <= END; SPLIT++))
  do
  echo "Split " $SPLIT
  python ./src/load_rgbd_washington_split.py $DATA_DIR $SPLIT_FILE $SPLIT
  done

