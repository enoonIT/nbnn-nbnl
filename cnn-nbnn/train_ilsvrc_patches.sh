#!/bin/bash

#CONF[1]="tag=full_batches_attempt2,batch=25600,save_every=750"
#CONF[1]="tag=2048_batch_oct1,batch=2048,save_every=8000,lambda=1,,nmz=std,n=30,passes=10"
CONF[1]="tag=full_batches_attempt4,batch=25600,save_every=750,lambda=1,nmz=std,dont-use-positions,n=20,passes=3"

PYTHON=/idiap/home/ikuzbor/code/l2sel/env/bin/python
cd /idiap/home/ikuzbor/code/cnn-nbnn/
$PYTHON eval.py do=train,${CONF[$SGE_TASK_ID]}
