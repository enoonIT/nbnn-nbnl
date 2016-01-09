
function data = load_patches(input_folder, patch_level)
    fprintf('Loading files from %s\n',input_folder);
    disp(input_folder)
    hdf5_files = dir(strcat(input_folder,'*.hdf5'));
    k=1;
    data = cell(numel(hdf5_files),1);
    for class_file = hdf5_files'
        class_file_path = strcat(input_folder, class_file.name);
        fprintf('Loading file %s\n', class_file_path);
        image_index = h5read(class_file_path,'/image_index');
        last = max(image_index(:));
        patches = h5read(class_file_path, patch_level);
        %positions = h5read(class_file_path,'/positions');
        %features = [patches(:,1:last); single(positions(:,1:last))];
        data{k} = matrixToCells(patches, image_index);
        k = k+1;
    end
end

