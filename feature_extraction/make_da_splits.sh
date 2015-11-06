#!/bin/bash
#jobs 0-59
source_d=(office/amazon office/amazon office/webcam office/webcam caltech10 caltech10)
target_d=(office/webcam caltech10 office/amazon caltech10 office/amazon office/webcam)
setting=(0 3)
locations=(unsuper super)
JOB_ID=$1
DATA_FOLDER=$2
OUT_FOLDER=$3
split=$((JOB_ID/12))
JOB_ID=$((JOB_ID % 12))
semi=$((JOB_ID/6))
loc=${locations[${semi}]}
tim=${setting[${semi}]}
JOB_ID=$((JOB_ID % 6))

POSTFIX=all_64_2_extra_hybrid_mean
source=${source_d[$JOB_ID]}
target=${target_d[$JOB_ID]}
SOURCE_DIR=${DATA_FOLDER}/${source_d[$JOB_ID]}/$POSTFIX
TARGET_DIR=${DATA_FOLDER}/${target_d[$JOB_ID]}/$POSTFIX
mods=${source//\//_}
modt=${target//\//_}
OUTPUT_DIR=${OUT_FOLDER}/${loc}/${mods}_${modt}
NUM_TRAIN=20    
NUM_TRANSF=$tim
SPLIT=$split


python ./src/da_splits.py --source-dir $SOURCE_DIR --target-dir $TARGET_DIR \
        --output-dir $OUTPUT_DIR --split $SPLIT \
       --train-images $NUM_TRAIN --transfer-imgs $NUM_TRANSF