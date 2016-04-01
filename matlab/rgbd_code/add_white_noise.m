function [ output_args ] = add_white_noise( filelist, input_dir, output_dir )
%ADD_WHITE_NOISE Summary of this function goes here
%   Detailed explanation goes here
    noise_intensity = 3;
    imagesize = 256;
    for i=1:numel(filelist)
        dark_light_bias = 20 - rand*40;
        img = double(imread(fullfile(input_dir, filelist{i})));
        noise = randn(imagesize/2);
        noise = imresize(noise, [256,256]);
        img = img + dark_light_bias + noise_intensity * noise;
        imwrite(uint8(img), fullfile(output_dir, filelist{i}));
        nholes = rand
    end

end
