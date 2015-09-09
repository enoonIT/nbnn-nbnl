#!/bin/bash

#
# This code extracts DECAF feature descriptors from patches of scene15 dataset.
# Configuration variables are set below (see extract_sports.sh for descriptions).
#
patch_size=(32 32 16)
levels=(3 3 4)
patch_method=(base extra extra)

DATA_DIR=$1
network_data_dir=$DATA_DIR/network/
i=$2
DATASET=caltech10
OUT_NAME=all_${patch_size[i]}_${levels[i]}_${patch_method[i]}_hybrid_mean;
OUTPUT_DIR=$DATA_DIR/desc/$DATASET/$OUT_NAME;



NUM_SPLITS=5
NUM_TRAIN=20	
NUM_TEST=-1
PATCHES_PER_IMAGE=100

rm $OUTPUT_DIR/relu -rf
rm $OUTPUT_DIR/nrelu -rf
echo $OUTPUT_DIR

python ./src/makeSplits.py --input-dir $OUTPUT_DIR --output-dir $OUTPUT_DIR/relu \
       --num-splits $NUM_SPLITS --patches $PATCHES_PER_IMAGE \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST --relu
       
python ./src/makeSplits.py --input-dir $OUTPUT_DIR --output-dir $OUTPUT_DIR/nrelu \
       --num-splits $NUM_SPLITS --patches $PATCHES_PER_IMAGE \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST
