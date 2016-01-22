function [D, V, PCAV] = learnCodebook(image_dir, img_list, pcadim, k)

num_files = min(1000, length(img_list));
rr = randperm(length(img_list));
img_list = img_list(rr);

% start the building
X = [];
for i=1:num_files
    name = img_list{i};
%     name=strrep(name,'.jpg','.mat');
%     name=strrep(name,'.JPEG','.mat');
    name = [image_dir, name, '.mat'];
    name = strtrim(name);
    fprintf('Loading image %s, %d/%d\n',name, i, num_files);
    try
        data = load(name);
        XX = double(data.S);
        XX(XX<0) = 0; % relu
    catch ME
        disp('wrong image')
        XX = [];
    end
    
    X = [X;XX];
end

C = cov(double(X));
[V,~] = eigs(C, pcadim);
X = double(X*V);

[~, D] = fkmeans(X, k);
%D = vl_kmeans(X, k);

if(length(img_list)>=5000)
	[X] = buildVLADALL(image_dir, img_list(1:5000), D, V, 1);
        [~,PCAV] = ScalPCA(X, 4096);
else
	[X] = buildVLADALL(image_dir, img_list, D, V, 1);
        [~,PCAV] = ScalPCA(X, 1000);
end





































