#!/bin/bash

#
# This code extracts DECAF feature descriptors from patches of scene15 dataset.
# Configuration variables are set below (see extract_sports.sh for descriptions).
# Usage ./extract_scene15.sh /input/data_dir jobId /destination/dir

patch_size=(16 32 64)
levels=(1 2 3)
#patch_method=(base extra extra)
COMBINATIONS=$((3)) #number of combinations for each dataset
DATA_DIR=$1
network_data_dir=$DATA_DIR/network/
i=$2
PATCH_EXTRACTION_METHOD=extra #${patch_method[i]}
DATASET=ISR67
PSIZE=${patch_size[$((i % COMBINATIONS))]}
LEVEL=${levels[$((i / COMBINATIONS))]}
PSIZE=16
LEVEL=3
INPUT_DIR=$DATA_DIR/images/$DATASET
OUT_NAME=all_${PSIZE}_${LEVEL}_${PATCH_EXTRACTION_METHOD}_hybrid_mean_dense
OUT_PARENT=$DATA_DIR
if [ "$#" -gt 2 ]; then
  OUT_PARENT=$3
fi
OUTPUT_DIR=${OUT_PARENT}/desc/$DATASET/$OUT_NAME

echo $OUTPUT_DIR

PATCHES_PER_IMAGE=100
IMAGE_DIM=200
DATA_SPLIT=-1 #-1 means all images, will ignore NUM_TEST and NUM_TRAIN
NUM_TRAIN=100 #100
NUM_TEST=100 #100
DECAF_LAYER_NAME=fc7_cudanet_out


for f in $INPUT_DIR/*; do
  echo $f
python ./src/extract.py --input-dir $f --output-dir $OUTPUT_DIR \
       --patches $PATCHES_PER_IMAGE --patch-size $PSIZE \
       --image-dim $IMAGE_DIM --descriptor DECAF \
       --levels $LEVEL --split $DATA_SPLIT \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST \
       --layer-name $DECAF_LAYER_NAME --network-data-dir $network_data_dir \
       --patch-method $PATCH_EXTRACTION_METHOD
done
