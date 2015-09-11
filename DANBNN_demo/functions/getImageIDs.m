function [ dataset_ids ] = getImageIDs( dir_name , allowedCategories)
%GETIMAGEIDS This method returns an array of cells which containes the
%number and ids of the contained images
%   Detailed explanation goes here
    files = dir(strcat(dir_name,'*.hdf5'));
    dataset_ids = cell(numel(files),1);
    x=1;
    for file = files'
        file_name = strcat(dir_name,file.name);
        image_index = h5read(file_name,'/image_index');
        nImages = size(image_index,2);
        dataset_ids{x}.data = [(1:nImages)' image_index']; %first column is image id, 2nd is first patch, 3d is last patch
        dataset_ids{x}.name = file.name;
        x = x+1;
    end
end

