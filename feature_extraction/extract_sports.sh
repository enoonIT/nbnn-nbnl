#!/bin/bash

#
# This code extracts DECAF feature descriptors from patches of sports dataset.
# Configuration variables with their descriptions are given below.
# Note that descriptors are extracted only from one data split.
#

# INPUT_DIR=data/images/sports                   # input directory of directories per each class, containing jpeg images
# OUTPUT_DIR=data/desc/sports                    # output directories to store descriptors, e.g. for training data: $OUTPUT_DIR/train/split_1/
# PATCES_PER_IMAGE=100                           # approximate number of patches per image to extract (usually extracted much less)
# PATCH_SIZE=64                                  # patch height/width in pixels
# IMAGE_DIM=200                                  # all images are resized to have this height, keeping aspect ratio
# LEVELS=3                                       # number of extraction pyramid levels
# DATA_SPLIT=1                                   # data split index
# NUM_TRAIN=70                                   # number of training images
# NUM_TEST=60                                    # number of testing images
# DECAF_LAYER_NAME=fc7_neuron_cudanet_out        # name of decaf layer (see decaf documentation for more info)

#!/bin/bash

#
# This code extracts DECAF feature descriptors from patches of scene15 dataset.
# Configuration variables are set below (see extract_sports.sh for descriptions).
# Note that descriptors are extracted only from one data split.
#

INPUT_DIR=data/images/sports
OUTPUT_DIR=data/desc/sports/all_2_64_extra_caffe7
PATCHES_PER_IMAGE=100
PATCH_SIZE=64
IMAGE_DIM=200
LEVELS=2
DATA_SPLIT=-1
NUM_TRAIN=100
NUM_TEST=100
DECAF_LAYER_NAME=fc7_cudanet_out

for f in $INPUT_DIR/*; do
#   if [ "$f" != "data/images/scene15/industrial" ]; then
#     continue
#   fi
  echo $f
python ./src/extract.py --input-dir $f --output-dir $OUTPUT_DIR \
       --patches $PATCHES_PER_IMAGE --patch-size $PATCH_SIZE \
       --image-dim $IMAGE_DIM --descriptor DECAF \
       --levels $LEVELS --split $DATA_SPLIT \
       --num-train-images $NUM_TRAIN --num-test-images $NUM_TEST \
       --layer-name $DECAF_LAYER_NAME
done


