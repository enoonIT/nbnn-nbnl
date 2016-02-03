function [ cov_m ] = incremental_cov( X, step_size )
%INCREMENTAL_COV This function computes the covariance matrix imcrementally
%   Usefull when n_samples >> n_dims
%   When n_dims >> n_samples is best to use randomized PCA
    samples = size(X,1);
    n_steps = ceil(samples / step_size);
    cov_m = zeros(size(X,2));
    at = 1;
    for s = 1:n_steps
        m = min(step_size, samples - (at-1));
        stepX = X(at:(at-1)+m,:);
        cov_m = cov_m + stepX'*stepX;
        at = at + m;
    end
    cov_m = (1/samples) * cov_m;
end

