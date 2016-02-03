function [xmean, xstd, COEFF, LATENT ] = getPCA_scalable( X , doSTD )
%GETABACUSFORMAT Summary of this function goes here
%   X should be samples x n_dims
    tic
    epsilon = 0.00001;
    [n m] = size(X);
    if doSTD
        disp 'Computing standardization (iterative)'
    	xmean = mean(X);
    	xstd = std(X);
        xstd(xstd==0)=epsilon; %to avoid numerical problems
        for k=1:n
            X(k, :) = ( X(k,:) - xmean ) ./ xstd; 
        end
    else
        disp 'Skipping standardization'
        xmean = zeros(1,m);
        xstd = ones(1,m);
    end
    
    disp 'Computing COVARIANCE matrix'
    sigma = incremental_cov(X, 10000);
    disp 'Computing SVD'
    [COEFF,S,V] = svd(sigma);
    LATENT = diag(S);
    toc
end

