function [xmean, xstd, COEFF, LATENT ] = getPCAMatrix( X )
%GETABACUSFORMAT Summary of this function goes here
%   X should be samples x n_dims
    tic
    [n m] = size(X);
    xmean = mean(X);
    xstd = std(X);
    B = (X - repmat(xmean,[n 1])) ./ repmat(xstd,[n 1]);
    [COEFF, SCORE, LATENT] = princomp(B);
    toc
end
