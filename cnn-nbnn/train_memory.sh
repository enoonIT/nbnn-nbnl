#!/bin/bash
train_images=80
test_images=20

python evalMy.py do=Train,input_folder=${1},train_images=${train_images},test_images=${test_images},tag=${2},batch=1024,passes=5
