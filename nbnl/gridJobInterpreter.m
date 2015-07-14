function [ lambda, input_folder, split ] = gridJobInterpreter( jobId )
%JOBINTERPRETER Summary of this function goes here
%   Detailed explanation goes here
    nLambda = 30;
    lambda_range =  linspace(1e-9,1e-5,nLambda);
    index = mod(jobId,nLambda);
    if index==0
        index = nLambda;
    end
    lambda = lambda_range(index);
    split = ceil(jobId/nLambda);
    input_folder = 'all_32_3_base_hybrid_mean';
end

