function [ dataset_ids ] = getImageIDs( dir_name , allowedCategories)
%GETIMAGEIDS This method returns an array of cells which containes the
%number and ids of the contained images
%   Detailed explanation goes here
    files = dir(strcat(dir_name,'*.hdf5'));
    dataset_ids = cell(1); %the actual number of classes depends on the allowed categories
    x=1;
    for file = files'
        filename = file.name;
        if(not(isempty(allowedCategories))) %if we are limiting ourselfs to certain categories
            if(any(ismember(allowedCategories, filename)))
                fprintf('Parsing: %s is an allowed category\n',filename);
            else
                fprintf('Skipping: %s is not an allowed category\n',filename);
                continue;
            end
        else
            fprintf('Parsing file %s...\n',filename);
        end
        file_name = strcat(dir_name,filename);
        image_index = h5read(file_name,'/image_index');
        nImages = size(image_index,2);
        dataset_ids{x}.data = [(1:nImages)' image_index']; %first column is image id, 2nd is first patch, 3d is last patch
        dataset_ids{x}.name = filename;
        x = x+1;
    end
    fprintf('Found %d valid categories for folder %s\n', numel(dataset_ids), dir_name);
end

