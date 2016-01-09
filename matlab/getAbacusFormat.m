function [ dataset ] = getAbacusFormat( input_folder, split_n )
%GETABACUSFORMAT Summary of this function goes here
%   Detailed explanation goes here
    train_folder = strcat(input_folder, '/train','/split_', num2str(split_n),'/');
    test_folder = strcat(input_folder, '/test','/split_', num2str(split_n),'/');
    dataset.train = load_patches(train_folder);
    dataset.test = load_patches(test_folder);
    save(strcat('dataset_split_',num2str(split_n)), 'dataset','-v7.3');
end
