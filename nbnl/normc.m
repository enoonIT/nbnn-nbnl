function [ norm_features ] = normc( features )
%NORMC Performs column-wise normalization of the matrix
%   Uses a for loop to reduce memory usage
    norm_features = zeros(size(features),'single');
    for col=1:size(features,2)
        column = features(:,col);
        norm_features(:,col) = column./sqrt(sum(column.^2));
    end
end

