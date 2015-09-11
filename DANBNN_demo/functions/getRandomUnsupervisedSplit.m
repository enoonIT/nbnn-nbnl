function [ testLabels testData trainData] = getRandomUnsupervisedSplit( SourceDataset, TargetDataset, trainingSamples)
%GETRANDOMUNSUPERVISEDSPLIT Summary of this function goes here
%   SourceDataset and TargetDataset .indexes are an array of cells containing the indexes and the
%   start and end patches for the source and target datasets. 
%   testLabels is an array containing the labels for the test set
%   testData is a cell array (one cell per image) containing the patches
%   trainData is a cell array (one cell per class) containing the patches
    Source = SourceDataset.indexes;
    Target = TargetDataset.indexes;
    isSameDomain = strcmp(SourceDataset.path, TargetDataset.path); % true if Source and Target are the same
    classes = numel(Source);
    trainData = cell(classes, 1);
    testLabels = [];
    testData = cell(1);
    for c=1:classes
        sourceDataset = strcat(SourceDataset.path, Source{c}.name); %absolute path to class hdf5 file
        indexes = Source{c}.data;
        shuffled = indexes(randperm(size(indexes,1)),:);
        trainId = shuffled(1:trainingSamples, :);
        trainData{c} = loadPatches(trainId, sourceDataset);
        if(isSameDomain)
            targetDataset = sourceDataset;
            testId = shuffled(trainingSamples+1:end, :);
        else
            assert(strcmp(Source{c}.name, Target{c}.name)) % make sure the categories are actually the same!
            targetDataset = strcat(TargetDataset.path, Target{c}.name);
            testId = Target{c}.data;
        end
    end
    
    
end

