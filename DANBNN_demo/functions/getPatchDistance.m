function [ delta ] = getPatchDistance(feat_tr, feat_te )
%GETPATCHDISTANCE Summary of this function goes here
%   feat_tr patches of class C (descriptorSize x nPatches)
%   feat_te patches of the image we want to compute the I2C distance
    build_params.algorithm='kdtree';
    K=1;
    [index_T, params_T] = flann_build_index(feat_tr, build_params);
    [ii,~]=flann_search(index_T, feat_te, K, params_T); %returns the indexes of the nearest patches for each test sample
    diff=feat_te-feat_tr(:,ii);
    delta=diff';
    flann_free_index(index_T);
end

