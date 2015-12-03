function [patches, labels, dataInfo, trainingMean] = getPatches(dataFolder, nImages)
    files = dir(strcat(dataFolder,'*.hdf5'));
    disp 'Extracting feature size information'
    class_count=1
    patches = []
    labels = []
    for file = files'
        file_name = strcat(dir_name,file.name);
        image_index = h5read(file_name,'/image_index');
        if(nImages ~= -1)
            shuffled = image_index(randperm(size(image_index,1)),:);
            image_index = shuffled(:, 1:nImages);
        end
        
        class_patches = loadPatches(image_index, file_name, false, false)
        patches = [patches class_patches];
        labels = [labels class_count*ones(1, size(class_patches,2))];
    end

%     if nargin < 4
%         disp('Computing scaling and centering factors.')
%         trainingMean = mean(features,2);
%     else
%         disp('Loading training scaling and centering factors')
%         trainingMean = oldMean;
%     end
%     disp('Centering vectors');
%     for i=1:size(features,2)
%         features(:,i) = features(:,i) - trainingMean;
%     end
%     disp('Normalizing vectors...')
%     features = normc(features);
end
