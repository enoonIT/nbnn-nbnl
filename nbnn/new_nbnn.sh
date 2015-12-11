#!/bin/bash

#
# This code tests NBNN classifier on a scenes15 dataset.
#

# DATA_DIR='/home/fmcarlucci/data/desc/scene15/all_32_3/splits'
DATA_DIR='/home/fmcarlucci/data/desc/sport8/all_32_3_extra_hybrid_mean/splits'
START=1
END=5

for ((SPLIT = START; SPLIT <= END; SPLIT++))
    do
    echo "Split " $SPLIT
    TRAIN_DIR="$DATA_DIR/train/split_$SPLIT/"
    TEST_DIR="$DATA_DIR/test/split_$SPLIT/"
    python src/nbnn.py $TRAIN_DIR $TEST_DIR -p --pca 256
    done

