function [ output_args ] = add_white_noise( filelist, input_dir, output_dir )
%ADD_WHITE_NOISE Summary of this function goes here
%   Detailed explanation goes here
    noise_intensity = 3;
    imagesize = 256;
    ih = imagesize / 2;
    hole_size = 17;
    n_imgs = numel(filelist);
    for i=1:n_imgs
        if mod(i, 1000) == 0
            fprintf('At %d/%d, elapsed %d secs\n',i,n_imgs, toc);
        end
        dark_light_bias = 20 - rand*40;
        img = double(imread(fullfile(input_dir, filelist{i})));
        noise = randn(imagesize/uint8(randsample([1 2 3], 1, true, [0.3 0.5 0.2])));
        noise = imresize(noise, [imagesize, imagesize]);
        img = img + dark_light_bias + noise_intensity * noise;
        nholes = randsample([0 1 2 3 4], 1, true, [0.2 0.5 0.2 0.06 0.04]);
        for h=1:nholes
            posX = uint8(rand*imagesize);
            posY = uint8(rand*imagesize);
            radius = hole_size + 3*randn;
            img = insertShape(img, 'FilledCircle', [posX, posY radius], 'Color', {'black'},'Opacity',1);
        end
        img = imrotate(img, rand*360);
        [wm hm] = size(img);
        wm = uint16(wm/2);
        hm = uint16(hm/2);
        img = img(1+wm-ih: wm+ih, 1+hm-ih:hm+ih);
        imwrite(uint8(img), fullfile(output_dir, filelist{i}));
    end

end
