#!/bin/bash

#
# This code creates training and testing splits from previously extraced image features
#  --relu sets all negative features to 0
#/mnt/WorkingDrive/data/desc/sport8/
# Usage /directory_name/dataset/ outfile_prefix
patch_sizes=(64)
levels=(1 2 3)
for patch_dim in ${patch_sizes[@]}; do
  for level in ${levels[@]}; do
    dir_name=${1}all_${patch_dim}_${level}_extra_hybrid_mean_dense/splits
    echo ${dir_name}_relu
    ./test_scene15.sh ${dir_name}_relu &> out/${2}_${patch_dim}_${level}_relu.out
        echo ${dir_name}
    ./test_scene15.sh ${dir_name} &> out/${2}_${patch_dim}_${level}.out
  done
done