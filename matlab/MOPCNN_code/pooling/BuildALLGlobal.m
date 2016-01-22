function [X] = BuildALLGlobal(image_dir, img_list)


% start the building
n_images = length(img_list);
X = zeros(n_images, 4096, 'single');
for i=1:n_images
    % load each data separately
    name = img_list{i};
%     name=strrep(name,'.jpg','.mat');
%     name=strrep(name,'.JPEG','.mat');
    name = [image_dir, name, '.mat'];
    name = strtrim(name);
    fprintf('Loading image %s, %d/%d\n',name, i, n_images);
    try
        v = load(name);
        v = v.S;
    catch ME
        disp('wrong image')
        v = randn(4096, 1);
        v(v<0)=0;
    end

    X(i,:) = v(:)';
end

