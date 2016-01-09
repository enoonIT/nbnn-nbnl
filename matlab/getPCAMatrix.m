function [ U ] = getPCAMatrix( X )
%GETABACUSFORMAT Summary of this function goes here
%   Detailed explanation goes here
    tic
    n_dims = size(X, 2)
    m=n_dims;
    Sigma = (1/m) * X' * X;
    [U,S,V] = svd(Sigma);
    toc
end
