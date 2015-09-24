function [ testLabels testData trainData trainIndexes] = getRandomUnsupervisedSplit( params)
%GETRANDOMUNSUPERVISEDSPLIT Summary of this function goes here
%   SourceDataset and TargetDataset .indexes are an array of cells containing the indexes and the
%   start and end patches for the source and target datasets. 
%   testLabels is an array containing the labels for the test set
%   testData is a cell array (one cell per image) containing the patches
%   trainData is a cell array (one cell per class) containing the patches
    SourceDataset = params.SourceDataset;
    TargetDataset = params.TargetDataset;
    relu = params.relu;
    trainingSamples = params.trainingSamples;
    Source = SourceDataset.indexes;
    Target = TargetDataset.indexes;
    isSameDomain = strcmp(SourceDataset.path, TargetDataset.path); % true if Source and Target are the same
    if(isSameDomain), disp 'Same domain',end;
    classes = numel(Source);
    trainData = cell(classes, 1);
    trainIndexes = cell(classes, 1);
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
        trainIndexes{c} = trainId;
        data = loadPatches(trainId, sourceDataset, relu);
        if(isfield(params, 'patchPercent'))
            prevSize = size(data,2);
            percent = params.patchPercent;
            idx = rand(1,size(data,2));
            keep = idx < percent;
            data = data(:,keep);
            fprintf('For class %d kept %2f of patch samples: from %d to %d\n',c, percent, prevSize, size(data,2));
        end
        trainData{c} = data;
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
        start = 1;
        if(params.supervised) %if supervised, add three images to training data
            for x=1:3
                patchesForSample = testId(x,3) - testId(x,2); 
                trainData{c} = [trainData{c} testDataTmp(:, firstPatch:firstPatch + patchesForSample-1)];
                firstPatch = firstPatch + patchesForSample;
                start = x+1;
            end
        end
        for idx=start:nSamples % assign the test patches to the corresponding test cell
            patchesForSample = testId(idx,3) - testId(idx,2); 
            testData{currentTestSample} = testDataTmp(:, firstPatch:firstPatch + patchesForSample-1);
            firstPatch = firstPatch + patchesForSample;
            currentTestSample = currentTestSample+1;
        end
        fprintf('Loaded %d test samples\n',currentTestSample-1);
        testLabels = [testLabels; ones((nSamples-start + 1),1)*c];
    end
    
    
end

