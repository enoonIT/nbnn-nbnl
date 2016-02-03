function [xmean, xstd, COEFF, LATENT ] = getPCAMatrix( X , doSTD )
%GETABACUSFORMAT Summary of this function goes here
%   X should be samples x n_dims
    tic
    epsilon = 0.00001;
    [n m] = size(X);
    if doSTD
	disp 'Computing standardization'
    	xmean = mean(X);
    	xstd = std(X);
    else
	disp 'Skipping standardization'
	xmean = zeros(1,m);
	xstd = ones(1,m);
    end
    xstd(xstd==0)=epsilon; %to avoid numerical problems
    X = (X - repmat(xmean,[n 1])) ./ repmat(xstd,[n 1]);
    disp 'Will now compute PCA'
    [COEFF, SCORE, LATENT] = princomp(X);
    toc
end
