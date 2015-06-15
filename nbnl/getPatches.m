function [features, labels, dataInfo, scalingFactor, trainingMean] = getPatches(dataset, split, test_or_train, oldScalingFactor, oldMean)
    size_cast = @double;
    dir_name = strcat('desc/',dataset, test_or_train,'/split_', num2str(split),'/');
    files = dir(strcat(dir_name,'*.hdf5'));
    features = size_cast([]);
    startPatch = 1;
    dataInfo = struct([]);
    k = 1;
    labels = [];
    disp 'Loading features'
    for file = files'
        file_name = strcat(dir_name,file.name);
        fprintf('Loading file: %s\n', file_name);
        image_index = h5read(file_name,'/image_index');
        lastPatch = max(max(image_index));
        patches = size_cast(h5read(file_name,'/patches'));
        positions = size_cast(h5read(file_name,'/positions'));
        features = [features [patches(:,1:lastPatch); positions(:,1:lastPatch)/20]];
        %features = [features patches(:,1:lastPatch)];
        labels = [labels; k * ones(lastPatch,1)];
        dataInfo(k).im_index = image_index;
        dataInfo(k).name = file_name;
        dataInfo(k).start = startPatch;
        startPatch = startPatch + lastPatch;
        dataInfo(k).end = startPatch - 1;
        k = k+1;
    end
    if nargin < 4
        disp('Computing scaling and centering factors.')
        scalingFactor = max(features,[],2);
        trainingMean = mean(features,2);
    else
        disp('Loading training scaling and centering factors')
        scalingFactor = oldScalingFactor;
        trainingMean = oldMean;
    end
%     scalingMatrix  = repmat( 1./scalingFactor, 1,  size(features,2) );
%     features = features .* scalingMatrix;
    disp('Centering vectors');
    me = repmat(trainingMean,1,size(features,2));
    features = features - me;
    clear me;
    disp('Normalizing vectors...')
    features = normc(features);
end
