function [ loaded_data ] = getRandomUnsupervisedSplit( params)
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
    testData = cell(0);
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
        data = loadPatchesDA(trainId, sourceDataset, relu, params.addPos, params.posScale);
        if(isfield(params, 'patchPercent'))
            prevSize = size(data,2);
            percent = params.patchPercent;
            idx = rand(1,size(data,2));
            keep = idx < percent;
            data = data(:,keep);
            fprintf('For class %d kept %.2f of patch samples: from %d to %d\n',c, percent, prevSize, size(data,2));
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
        testDataTmp = loadPatchesDA(testId, targetDataset, relu, params.addPos, params.posScale); % load all the test patches
        firstPatch = 1;
        classTestData = cell(1,nSamples);
        currentTestSample = 1;
        for idx=1:nSamples % assign the test patches to the corresponding test cell
            patchesForSample = testId(idx,3) - testId(idx,2); 
            classTestData{currentTestSample} = testDataTmp(:, firstPatch:firstPatch + patchesForSample-1);
            firstPatch = firstPatch + patchesForSample;
            currentTestSample = currentTestSample+1;
        end
        targetImagesToTransfer = 3;
        if(params.supervised) %if supervised, add three images to training data //TODO fix the fact that always same images are added
            shuffledIndexes = randperm(numel(classTestData));
            toAdd = shuffledIndexes(1:targetImagesToTransfer);
            for i=toAdd
                trainData{c} = [trainData{c} classTestData{i}];
            end
            classTestData(toAdd) = [];
            fprintf('Test images %d added to Source\n',toAdd);
        end
        testData = [testData classTestData];
        fprintf('Loaded %d test samples\n',currentTestSample-1);
        testLabels = [testLabels; ones(numel(classTestData),1)*c];
    end
    loaded_data.testLabels =testLabels;
    loaded_data.testData = testData;
    loaded_data.trainData = trainData;
    loaded_data.trainIndexes = trainIndexes;
end

