function resultMatrix = getResultsInTable()
    jobs = 144;
    resultMatrix = zeros(36,8); % 4 columns for acc and 4 for std
    patchs = [16 32 64];
    levels = [1 2 3];
    datasets = {'office/amazon' 'office/webcam' 'office/dslr' 'caltech10'};
    for i=1:144
        filename = strcat('job_NBNN_Relu_',num2str(i),'.mat');
        if(not(exist(filename,'file')))
            return
        end
        T = load(filename);
        row = (T.params.levels-1)*12 + (find(patchs==T.params.patchSize)-1)*4 + find(strcmp(datasets, T.params.SourceDataset.dataset));
        col = 1 + (find(strcmp(datasets, T.params.TargetDataset.dataset)) -1)*2;
        resultMatrix(row,col) = mean(T.accuracy);
        resultMatrix(row,col+1) = std(T.accuracy);
    end
end