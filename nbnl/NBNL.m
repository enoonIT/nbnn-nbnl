function [ confusion, accuracy ] = NBNL( dec_values, dataInfo )
%NBNL run NBNL on the patches that have been scored by ML3
%   Runs NBNL and returns a confusion matrix and the accuracy
    tic
    start = 1;
    classes = size(dataInfo,2)
    confusion = zeros(classes);
    for class = 1:classes
        image_index = dataInfo(class).im_index;
        newStart = 0;
        for image = image_index
            values = dec_values(start + image(1): start + image(2) - 1, :);
            newStart = start + image(2);
            s = sum(values);
            [~, pred_class] = max(s);
            confusion(class, pred_class) = confusion(class, pred_class) +1;
        end
        start = newStart;
        
    end
    accuracy = trace(confusion)/sum(confusion(:));
    toc
end

