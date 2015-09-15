function [ testLabels testData trainData] = getRandomUnsupervisedSplit( SourceDataset, TargetDataset, trainingSamples, relu)
%GETRANDOMUNSUPERVISEDSPLIT Summary of this function goes here
%   SourceDataset and TargetDataset .indexes are an array of cells containing the indexes and the
%   start and end patches for the source and target datasets. 
%   testLabels is an array containing the labels for the test set
%   testData is a cell array (one cell per image) containing the patches
%   trainData is a cell array (one cell per class) containing the patches
    Source = SourceDataset.indexes;
    Target = TargetDataset.indexes;
    isSameDomain = strcmp(SourceDataset.path, TargetDataset.path); % true if Source and Target are the same
    if(isSameDomain), disp 'Same domain',end;
    classes = numel(Source);
    trainData = cell(classes, 1);
    testLabels = [];
    testData = cell(1);
    currentTestSample = 1;
    for c=1:classes
        fprintf('Loading test and train for class %d %s\n',c, Source{c}.name);
        sourceDataset = strcat(SourceDataset.path, Source{c}.name); %absolute path to class hdf5 file
        indexes = Source{c}.data;
        shuffled = indexes(randperm(size(indexes,1)),:);
        nTrainSamples = trainingSamples;
        nImages = size(shuffled,1);
        if(nImages<=(trainingSamples+1)) % this is for the classes with less than a certain number of training samples
            if(isSameDomain)
                nTrainSamples = floor(nImages/2);
            else
                nTrainSamples = nImages;
            end
            fprintf('Class contains %d images, setting training sample size to %d\n',nImages,nTrainSamples);
        end
        trainId = shuffled(1:nTrainSamples, :);
        trainData{c} = loadPatches(trainId, sourceDataset, relu);
        if(isSameDomain)
            targetDataset = sourceDataset;
            testId = shuffled(nTrainSamples+1:end, :);
        else
            assert(strcmp(Source{c}.name, Target{c}.name)) % make sure the categories are actually the same!
            targetDataset = strcat(TargetDataset.path, Target{c}.name);
            testId = Target{c}.data;
        end
        % load test data
        nSamples = size(testId,1);
        testDataTmp = loadPatches(testId, targetDataset, relu); % load all the test patches
        firstPatch = 1;
        for idx=1:nSamples % assign the test patches to the corresponding test cell
            patchesForSample = testId(idx,3) - testId(idx,2); 
            testData{currentTestSample} = testDataTmp(:, firstPatch:firstPatch + patchesForSample-1);
            firstPatch = firstPatch + patchesForSample;
            currentTestSample = currentTestSample+1;
        end
        fprintf('Loaded %d test samples\n',currentTestSample-1);
        testLabels = [testLabels; ones(nSamples,1)*c];
    end
    
    
end

