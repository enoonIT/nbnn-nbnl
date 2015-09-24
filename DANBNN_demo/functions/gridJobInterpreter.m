function [ params ] = gridJobInterpreter( jobId , dataDir)
%GRIDJOBINTERPRETER Summary of this function goes here
%   we have 2 patch size configuration and 4 source and target datasets -
%   and 10 steps for training size, 2 for relu
    % get algorithm parameters
    params.relu = false;
    categories = {'backpack.hdf5' 'headphones.hdf5' 'monitor.hdf5' 'bike.hdf5' 'keyboard.hdf5' 'mouse.hdf5' 'projector.hdf5' 'calculator.hdf5'  'laptop.hdf5' 'mug.hdf5'};
    params.categories = categories;
    if(jobId>320)
        params.relu = true;
        jobId = jobId - 320;
    end

    BLOCK = 32; % 2 patch size * 4 source * 4 target
    patch_conf=[3 32; 2 64];
    datasets = {'office/amazon' 'office/webcam' 'office/dslr' 'caltech10'};
    
    trainMalus = ceil(jobId / BLOCK); %1-10
    jobMod = mod(jobId, BLOCK); 
    i = find(jobMod==0); jobMod(i)=BLOCK; %1-32
    MINIBLOCK = 16; % 4 source and 4 target
    patchSet = ceil(jobMod/MINIBLOCK);
    patch = patch_conf(patchSet,2);
    level = patch_conf(patchSet,1);
    jobMod = mod(jobMod, MINIBLOCK);
    i = find(jobMod==0); jobMod(i)=MINIBLOCK;
    MINIBLOCK = 4;
    sourceD = datasets{ceil(jobMod/MINIBLOCK)};
    jobMod = mod(jobMod, MINIBLOCK);
    i = find(jobMod==0); jobMod(i)=MINIBLOCK;
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
    params.supervised = true;
    params.splits = 10;
    params.patchPercent = trainMalus/10.0;
    fprintf('%d %s - %s -> %s - - - %d: %.2f%%\n',jobId, folderName,sourceD,targetD, trainingSamples, params.patchPercent*100);
end

