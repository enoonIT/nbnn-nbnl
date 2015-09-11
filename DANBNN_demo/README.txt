
This README explains how to use the demo for the DA-NBNN algorithm presented 
in the ICCV 2013 paper "Frustratingly Easy NBNN Domain Adaptation".

The code needs two tools:

1) FLANN (Fast Library for Approximate Nearest Neighbors)
You can download the code from
http://www.cs.ubc.ca/research/flann/uploads/FLANN/flann-1.8.4-src.zip
and copy all the files in "flann-1.8.4-src/src/matlab" inside the directory 
"flann".

2) k Nearest Neighbor
Our code uses the funcion kNearestNeighbors(dataMatrix, queryMatrix, k)
that can be downloaded from 
http://www.mathworks.com/matlabcentral/fileexchange/15562-k-nearest-neighbors
and should be copied in the directory "functions".

Run the following matlab script

>> demo

The demo runs the Caltech->Amazon experiment described in Section 5.2 (and
Figure 3) of the paper, together with its corresponding in-domain run on
Amazon->Amazon. It gives as output the BOW-Nearest Neighbor, NBNN and DA-NBNN 
results.

The code starts by defining a split over the Caltech and Amazon data (the
corresponding lines are commented here and a single set is pre-calculated).

step 1: DA-NBNN is initialized with the results of NBNN
step 2: 20 samples are removed from the source training set, and 20 samples 
are added from the target test set to the target training set. In both cases
2 samples are chosen from each of the 10 classes.

The full DA-NBNN method requires multiple subsequent iterations of step
2, but from the results it is clear that even after the first one we get
an improvement in recognition rate with respect to NBNN.

Below there is a list of the remaining content of this folder.


Office+Caltech/

This directory contains the data files of the Amazon and Caltech domain
from the Office+Caltech dataset.
We extracted SURF features from the original Office+Caltech images by using OpenSURF
(http://www.mathworks.it/matlabcentral/fileexchange/28300-opensurf-including-image-warp).

Office+Caltech/amazon_vocabulary_BOW/

A 800-visual-word BOW vocabulary was obtained by using k-means over a random 
selection of the Amazon images. All the images are then represented by using 
this codebook as reference.

splits/

The split mat-files are saved in this directory. We separate each domain 
in a training set with 20 samples per class and a test set containing
all the remaining images.

flann/

Our demo uses FLANN both for NBNN and for DA-NBNN. The matlab FLANN source 
files should be in this directory. 
We provide the mex file for nearest_neighbors.cpp. This file was compiled 
using MATLAB 8.1.0.604 (R2013a) under 64 bits Linux environment. In case of 
a different architecture you need to recompile it.

functions/

In this directory there are all the main functions necessary to run the
demo.

*select.m
It runs the sample selection for each domain and save the sample indices
as mat files in the /splits directory.

*run_NN.m
It runs the BOW-NN experiments, both in- and cross-domain.

*run_NBNN.m
It runs NBNN for the in-domain setting.

*run_DA-NBNN.m
It starts by running NBNN for the cross-domain setting and then continues
with DA-NBNN. Only one step of sample selection + matrix optimization is 
considered.

*adaptation.m
*fn_create_dist.m
*fn_create_metric.m
*add.m
Functions used by the main DA-NBNN code.


OUTPUT EXAMPLE (obtained with Matlab R2013a) ------------------------------

>> demo

BOW-NN Amazon->Amazon, rec. rate: 44.46 %
BOW-NN Caltech->Amazon, rec. rate: 23.75 %

Testing NBNN on class 1, with K=1...
Testing NBNN on class 2, with K=1...
Testing NBNN on class 3, with K=1...
Testing NBNN on class 4, with K=1...
Testing NBNN on class 5, with K=1...
Testing NBNN on class 6, with K=1...
Testing NBNN on class 7, with K=1...
Testing NBNN on class 8, with K=1...
Testing NBNN on class 9, with K=1...
Testing NBNN on class 10, with K=1...

NBNN Amazon->Amazon, rec. rate: 63.32 %

Calculate distances...
Testing NBNN on class 1, with K=1...
Testing NBNN on class 2, with K=1...
Testing NBNN on class 3, with K=1...
Testing NBNN on class 4, with K=1...
Testing NBNN on class 5, with K=1...
Testing NBNN on class 6, with K=1...
Testing NBNN on class 7, with K=1...
Testing NBNN on class 8, with K=1...
Testing NBNN on class 9, with K=1...
Testing NBNN on class 10, with K=1...

step 1, NBNN Caltech->Amazon, rec. rate: 40.77 %

Calculate distances...
Metric optimization.........
Testing NBNN on class 1, with K=1...
Testing NBNN on class 2, with K=1...
Testing NBNN on class 3, with K=1...
Testing NBNN on class 4, with K=1...
Testing NBNN on class 5, with K=1...
Testing NBNN on class 6, with K=1...
Testing NBNN on class 7, with K=1...
Testing NBNN on class 8, with K=1...
Testing NBNN on class 9, with K=1...
Testing NBNN on class 10, with K=1...

step 2, DA-NBNN Caltech->Amazon, rec. rate: 50.79 %

