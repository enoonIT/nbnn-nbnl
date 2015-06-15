#!/bin/bash

#
# This code creates training and testing splits from previously extraced image features
#  --relu sets all negative features to 0

#INPUT_DIR=data/desc/sports/all_1_16_caffe7	
INPUT_DIR=data/desc/scene15/train/split_5

python ./src/FeatureChecker.py --input-dir $INPUT_DIR
	  