function [ data ] = transform( patches, dataInfo )
%TRANSFORM Summary of this function goes here
    start = 1;
    classes = size(dataInfo,2);
    data = cell(classes, 100);
    for class = 1:classes
        iCount = 1;
        image_index = dataInfo(class).im_index;
        newStart = 0;
        for image = image_index
            im_patches = patches(:, start + image(1):start+image(2)-1);
            newStart = start + image(2);
            data{class, iCount} = im_patches;
            iCount = iCount +1;
        end
        start = newStart;
        
    end
end

