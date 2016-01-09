function [xmean, xstd, COEFF, LATENT ] = getPCAMatrix( X )
%GETABACUSFORMAT Summary of this function goes here
%   X should be samples x n_dims
    tic
    epsilon = 0.00001;
    [n m] = size(X);
    xmean = mean(X);
    xstd = std(X);
    xstd(xstd==0)=epsilon; %to avoid numerical problems
    B = (X - repmat(xmean,[n 1])) ./ repmat(xstd,[n 1]);
    [COEFF, SCORE, LATENT] = princomp(B);
    toc
end
