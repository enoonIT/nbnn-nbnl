function [p, lambda, input_folder, split ] = gridJobInterpreter( jobId )
%JOBINTERPRETER Summary of this function goes here
%   Detailed explanation goes here
    ps = [1:0.1:2 3 10 1000];
    lambdas = [1e-07 1e-06 1e-05 1e-04];
    nP = length(ps);
    nLambda = length(lambdas);
    blockSize = nP*nLambda;
    
    index = mod(jobId, blockSize);
    if index==0
        index = blockSize;
    end
    [pI lI] = ind2sub([nP nLambda],index);
    lambda = lambdas(lI);
    p = ps(pI);
    split = ceil(jobId/blockSize);
    input_folder = 'all_32_3_base_hybrid_mean';
end

