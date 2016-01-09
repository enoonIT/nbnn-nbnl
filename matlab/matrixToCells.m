function cell_data = matrixToCells(features, iid)
    num_images = size(iid,2);
    cell_data = cell(1, num_images);
    for x=1:num_images
        idx = iid(:,x);
        cell_data{x} = features(:, idx(1)+1:idx(2));
    end
end

