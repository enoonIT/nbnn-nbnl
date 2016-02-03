clear
LEVEL=7;
patch=strcat('/patches',num2str(LEVEL));

datasets{1}.data = '/home/fmcarlucci/data/desc/ART/test/';
datasets{1}.name = 'art_test';
datasets{2}.data = '/home/fmcarlucci/data/desc/ART/validation/';
datasets{2}.name = 'art_validation';
datasets{3}.data = '/home/fmcarlucci/data/desc/ART/train/';
datasets{3}.name = 'art_train';
datasets{4}.data = '~/data/desc/VOC/all_32_3_hybrid_mean/';
datasets{4}.name = 'VOC';
elements=1:numel(datasets);
RELU=false;
STANDARDIZE=true;
DIMS=128;
srelu='norelu';
if RELU, srelu='relu';end
sstd='nostd';
if STANDARDIZE, sstd='std';end

dir_name = strcat('dims_',num2str(DIMS),'_',sstd ,'_', srelu,'_level',num2str(LEVEL),'/');
if not(exist(dir_name,'dir'))
	mkdir(dir_name);
end
name = 'ART';
fprintf('Start cycle for target %s\n', name)
X = cell_to_matrix(load_patches(datasets{1}.data, patch));
l = randperm(size(X,1));
X = X( l(1: round(size(X,1)/2)), :);
X2 = cell_to_matrix(load_patches(datasets{2}.data, patch));
l = randperm(size(X2,1));
X = [X; X2( l(1: round(size(X2,1)/2)), :)];
X2 = cell_to_matrix(load_patches(datasets{3}.data, patch));
l = randperm(size(X2,1));
X = [X; X2( l(1: round(size(X2,1)/2)), :)];
clear X2

to_apply = elements;
if RELU
    disp 'Applying RELU'
    X(X<0)=0;
end
[xmean, xstd, coeff, latent ] = getPCAMatrix(X, STANDARDIZE);
variance_kept = sum(latent(1:DIMS))/sum(latent)
disp 'Applying PCA to targets'
for dest=to_apply
    fprintf('Applying to %s\n', datasets{dest}.name)
    reduced_data = apply_pca(datasets{dest}.data, coeff, DIMS, xmean, xstd, RELU);
    save(strcat(dir_name, name,'_to_',datasets{dest}.name), 'reduced_data');
end
save(strcat(dir_name, 'variance_', name), 'variance_kept');
