function [ ] = convert_16to8( input_file , outpath)
%CONVERT_16TO8 Summary of this function goes here
%   Detailed explanation goes here
    co = 0.128; %255 / 2000;
    img = imread(input_file);
    img = double(img) * co;
    img(img>255)=255;
    imwrite(uint8(img), outpath);
end