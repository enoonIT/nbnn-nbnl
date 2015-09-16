function [] = expressResults( id )
%EXPRESSRESULTS Summary of this function goes here
%   Detailed explanation goes here
    filename = strcat('job_NBNN_Relu_',num2str(id),'.mat');
    if(not(exist(filename)))
        return
    end
    T = load(filename);
    fprintf('%3d: %s -> %s [%d %d]  Acc: %f Std: %f\n',id, T.params.SourceDataset.dataset,T.params.TargetDataset.dataset,T.params.patchSize, T.params.levels, mean(T.accuracy),std(T.accuracy));

end

