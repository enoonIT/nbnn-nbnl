function [ Ured ] = getPCATransform( data )
%GETPCATRANSFORM Summary of this function goes here
%   Detailed explanation goes here
    patchCount = size(data,2);
    Sigma = (1/patchCount) * data * data';
    [U,S,V] = svd(Sigma);
    y = zeros(1,patchCount);
    for x=1:size(S,2)
        y(x) = y(x) + S(x,x);
        y(x+1) = y(x);
    end
    Y = max(y);
    K = Y*0.99;
    i = find(y>K,1);
    fprintf('PCA reduction to %d dimensions\n',i);
    Ured = U(:,1:i);
end

