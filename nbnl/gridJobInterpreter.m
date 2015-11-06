function [algM, input_folder, split ] = gridJobInterpreter( jobId , data_folder)
%JOBINTERPRETER Summary of this function goes here
%   Detailed explanation goes here
    source_d={'office_amazon', 'office_amazon', 'office_webcam', 'office_webcam', 'caltech10', 'caltech10'};
    target_d={'office_webcam', 'caltech10', 'office_amazon', 'caltech10', 'office_amazon', 'office_webcam'};
    setting=[0 3];
    locations={'/unsuper/', '/super/'};
    split=ceil(jobId/12)-1;
    jobId=mod(jobId, 12);
    jobId(jobId==0)=12;
    semi=ceil(jobId/6);
    loc=char(locations(semi));
    tim=setting(semi);
    jobId=mod(jobId, 6);
    jobId(jobId==0)=6;
    source=char(source_d(jobId));
    target=char(target_d(jobId));
    algM = 10;
    input_folder =strcat(data_folder,loc,source,'_',target,'/');
    fprintf('%s\n',input_folder);
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
end

