function [ params ] = gridJobInterpreter( jobId , dataDir)
%GRIDJOBINTERPRETER Summary of this function goes here
%   we have 2 patch size configuration and 4 source and target datasets -
%   and 10 steps for training size, 2 for relu
    % get algorithm parameters
    params.relu = false;
    categories = {'backpack.hdf5' 'headphones.hdf5' 'monitor.hdf5' 'bike.hdf5' 'keyboard.hdf5' 'mouse.hdf5' 'projector.hdf5' 'calculator.hdf5'  'laptop.hdf5' 'mug.hdf5'};
    params.categories = categories;
    datasets = {'office/amazon' 'office/webcam' 'office/dslr', 'caltech10'};
    level = 2;
    patch = 64;
    jobMod = jobId;
    MINIBLOCK = numel(datasets);
    sourceD = datasets{ceil(jobMod/MINIBLOCK)};
    jobMod = mod(jobMod, MINIBLOCK);
    jobMod(jobMod==0)=MINIBLOCK;
    targetD = datasets{jobMod};
    % prepare params object
    folderName = strcat('/all_',num2str(patch),'_',num2str(level),'_extra_hybrid_mean/');
    S.path = strcat(dataDir,'/desc/',sourceD,folderName);
    S.dataset = sourceD;
    S.indexes = getImageIDs(S.path, categories);
    T.path = strcat(dataDir,'/desc/',targetD,folderName);
    T.dataset = targetD;
    T.indexes = getImageIDs(T.path, categories);
    params.SourceDataset = S;
    params.TargetDataset = T;
    trainingSamples = 20;
    if(strcmp(targetD,'office/dslr') || strcmp(targetD,'office/webcam'))
        trainingSamples = 15;
    end
    params.trainingSamples = trainingSamples;
    params.patchSize = patch;
    params.levels = level;
    params.supervised = false;
    params.splits = 5;
    params.addPos = false;
    params.posScale = 0.1;
    fprintf('%d %s - %s -> %s - - - %d: (Relu %s) %.2f%%\n',jobId, folderName,sourceD,targetD, trainingSamples, char(params.relu+48), 100);%params.patchPercent*100);
end

