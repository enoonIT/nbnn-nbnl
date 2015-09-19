function [ params ] = gridJobInterpreter( jobId , dataDir)
%GRIDJOBINTERPRETER Summary of this function goes here
%   we have 9 patch size configuration and 4 source and target datasets -
%   we have 144 possible jobs
    % get algorithm parameters
    params.relu = false;
    categories = {'backpack.hdf5' 'headphones.hdf5' 'monitor.hdf5' 'bike.hdf5' 'keyboard.hdf5' 'mouse.hdf5' 'projector.hdf5' 'calculator.hdf5'  'laptop.hdf5' 'mug.hdf5'};
    params.categories = categories;
%     if(jobId>144)
%         %params.relu = true;
%         jobId = jobId - 144;
%     end
    jobId = mod(jobId, 144); if(jobId==0)jobId=144;end
    BLOCK = 48; % 3 patch size * 4 source * 4 target
    patch_size=[16 32 64];
    levels=[1 2 3];
    datasets = {'office/amazon' 'office/webcam' 'office/dslr' 'caltech10'};
    level = levels(ceil(jobId/BLOCK));
    jobMod = mod(jobId, BLOCK);
    i = find(jobMod==0); jobMod(i)=BLOCK;
    MINIBLOCK = 16; % 4 source and 4 target
    patch = patch_size(ceil(jobMod/MINIBLOCK));
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
    params.supervised = false;
    fprintf('%d %s - %s -> %s - - - %d\n',jobId, folderName,sourceD,targetD, trainingSamples);
end

