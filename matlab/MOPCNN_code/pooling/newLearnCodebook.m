function [D, V, PCAV] = newLearnCodebook(img_list, pcadim, k, lvl)

num_files = min(1000, length(img_list));
rr = randperm(length(img_list));
img_list = img_list(rr);

% start the building
X = [];
for i=1:num_files
    name = img_list{i};
    fprintf('Loading image %s, %d/%d\n',name, i, num_files);
    try
        data = h5read(name, '/feats');  
	level = h5read(name, '/level');
        XX = double(data(:, level==lvl))';
%        XX(XX<0) = 0; % relu
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
	[X] = newBuildVLADALL(img_list(1:5000), D, V, 1, lvl);
        [~,PCAV] = ScalPCA(X, 4096);
else
	[X] = newBuildVLADALL(img_list, D, V, 1, lvl);
        [~,PCAV] = ScalPCA(X, 1000);
end





































