function [ lambda, input_folder ] = jobInterpreter( jobId )
%JOBINTERPRETER Summary of this function goes here
%   Detailed explanation goes here
    lambda_range =  5*(1.5.^(1:30)./10^6);
    nLambda = length(lambda_range);
    index = mod(jobId,nLambda);
    if index==0
        index = nLambda;
    end
    lambda = lambda_range(index);
    data_folders = {'all_16_4_extra_hybrid_mean', 'all_32_3_base_hybrid_mean','all_32_3_extra_hybrid_mean'};
    dataIndex = ceil(jobId/nLambda);
    input_folder = char(data_folders(dataIndex));
end

