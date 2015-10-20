#!/bin/bash

#
# This code creates training and testing splits from previously extraced image features
#  --relu sets all negative features to 0
# Usage ./autoSplits.sh /data/desc/dataset/
patch_sizes=(16 32 64)
levels=(1 2 3)
for patch_dim in ${patch_sizes[@]}; do
  for level in ${levels[@]}; do
    dir_name=${1}/all_${patch_dim}_${level}_extra_hybrid_mean/
    echo ${dir_name} relu
    ./splitHDF5.sh ${dir_name} --relu
    echo ${dir_name} not relu
    ./splitHDF5.sh ${dir_name}
  done
done