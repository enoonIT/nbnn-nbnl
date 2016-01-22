#!/bin/bash

#
# This code extracts CNN feature descriptors from a given dataset.
# Configuration variables are set below
#

DATA_DIR=$1
network_data_dir=$DATA_DIR/network/
PATCH_EXTRACTION_METHOD=extra #${patch_method[i]}
DATASET=ART
PSIZE=32
LEVEL=3
INPUT_DIR=$2
OUT_NAME=all_${PSIZE}_${LEVEL}_hybrid_mean
OUTPUT_DIR=$DATA_DIR/desc/$DATASET/$OUT_NAME

echo $OUTPUT_DIR

PATCHES_PER_IMAGE=100
IMAGE_DIM=200
DATA_SPLIT=-1 #-1 means all images, will ignore NUM_TEST and NUM_TRAIN
NUM_TRAIN=100 #100
NUM_TEST=100 #100
DECAF_LAYER_NAME=fc7_cudanet_out

OUTPUT_DIR=$DATA_DIR/desc/$DATASET/test/$OUT_NAME
for f in $INPUT_DIR/test/*; do
  echo $f
python ./src/extract.py --input-dir $f --output-dir $OUTPUT_DIR \
       --patches $PATCHES_PER_IMAGE --patch-size $PSIZE \
       --image-dim $IMAGE_DIM --descriptor DECAF \
       --levels $LEVEL --split $DATA_SPLIT \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST \
       --layer-name $DECAF_LAYER_NAME --network-data-dir $network_data_dir \
       --patch-method $PATCH_EXTRACTION_METHOD
done
OUTPUT_DIR=$DATA_DIR/desc/$DATASET/train/$OUT_NAME
for f in $INPUT_DIR/train/*; do
  echo $f
python ./src/extract.py --input-dir $f --output-dir $OUTPUT_DIR \
       --patches $PATCHES_PER_IMAGE --patch-size $PSIZE \
       --image-dim $IMAGE_DIM --descriptor DECAF \
       --levels $LEVEL --split $DATA_SPLIT \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST \
       --layer-name $DECAF_LAYER_NAME --network-data-dir $network_data_dir \
       --patch-method $PATCH_EXTRACTION_METHOD
done
OUTPUT_DIR=$DATA_DIR/desc/$DATASET/validation/$OUT_NAME
for f in $INPUT_DIR/validation/*; do
  echo $f
python ./src/extract.py --input-dir $f --output-dir $OUTPUT_DIR \
       --patches $PATCHES_PER_IMAGE --patch-size $PSIZE \
       --image-dim $IMAGE_DIM --descriptor DECAF \
       --levels $LEVEL --split $DATA_SPLIT \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST \
       --layer-name $DECAF_LAYER_NAME --network-data-dir $network_data_dir \
       --patch-method $PATCH_EXTRACTION_METHOD
done