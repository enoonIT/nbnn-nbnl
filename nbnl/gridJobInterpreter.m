function [algM, input_folder, split ] = gridJobInterpreter( jobId , data_folder)
%JOBINTERPRETER Summary of this function goes here
%   Detailed explanation goes here
%     test_datasets = {'all_64_2_extra_hybrid_mean'};
%     algMs = [3 10 30];
%     combinations = 2*5; % 2 patch settings, 5 splits
%     algM = algMs(ceil(jobId/combinations));
%     jobId = mod(jobId, combinations);
%     jobId(jobId==0)=combinations;
%     jobD = ceil(jobId/5);
%     input_folder = strcat(data_folder, '/desc/ISR67/',test_datasets{jobD}, '/splits/');
% %     ps = [1:0.1:2 3 10 1000];
% %     lambdas = [1e-07 1e-06 1e-05 1e-04];
% %     nP = length(ps);
% %     nLambda = length(lambdas);
% %     blockSize = nP*nLambda;
%     
%     split = mod(jobId, 5);
%     split(split==0)=5;
% %     [pI lI] = ind2sub([nP nLambda],index);
% %     lambda = lambdas(lI);
% %     p = ps(pI);
% %     split = ceil(jobId/blockSize);
%     fprintf('%d %s split: %d\n',algM,input_folder,split);
    algM = 10
    input_folder = strcat(data_folder, '/desc/webcam_amazon/');
    split = 1
end

