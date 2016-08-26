function [X] = buildVLADALL(img_list, D, V, PCAV, lvl)

D = single(D);
% start the building
X = zeros(length(img_list), size(D,1)*size(V,2), 'single');
for i=1:length(img_list)
    % load each data separately
    name = img_list{i};
    if mod(i,500)==0
        fprintf('Loading image %s, %d/%d\n',name, i, length(img_list));
    end

    try
       data = h5read(name, '/feats');
       level = h5read(name, '/level');
       XX = double(data(:, level==lvl))';
%        XX(XX<0) = 0; % relu
       XX = XX*V;
       [v] = FisherVector(XX, D);
       v = v';
    catch ME
        disp('wrong image')
        v = randn(1,size(D,1)*size(V,2));
    end

    X(i,:) = v(:);
end

X = sign(X) .* sqrt(abs(X));
X = normalize(X);

X = X*PCAV;




















