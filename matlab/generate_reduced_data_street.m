clear
LEVEL=7;
patch=strcat('/patches',num2str(LEVEL));
datasets{1}.datapath = '~/data/desc/Street2Shop/test/street/';
datasets{1}.name = 'street_test';
datasets{2}.datapath = '~/data/desc/Street2Shop/train/street/';
datasets{2}.name = 'street_train';
datasets{3}.datapath = '~/data/desc/Street2Shop/test/product/';
datasets{3}.name = 'product_test';
datasets{4}.datapath = '~/data/desc/Street2Shop/train/product/';
datasets{4}.name = 'product_train';
targets=[1,3];

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
for idx=targets
	name = datasets{idx}.name;
	fprintf('Start cycle for target %s\n', name)
	X = cell_to_matrix(load_patches(datasets{idx}.datapath, patch));
	if RELU
		disp 'Applying RELU'
		X(X<0)=0;
	end
	[xmean, xstd, coeff, latent ] = getPCA_scalable(X, STANDARDIZE);
	variance_kept = sum(latent(1:DIMS))/sum(latent)
	disp 'Applying PCA to targets'
    clear X
	for dest=1:numel(datasets)
		fprintf('Applying to %s\n', datasets{dest}.name)
		reduced_data = apply_pca(load_patches(datasets{dest}.datapath, patch), coeff, DIMS, xmean, xstd, RELU);
		save(strcat(dir_name, name,'_to_',datasets{dest}.name), 'reduced_data');
	end
	save(strcat(dir_name, 'variance_', name), 'variance_kept');
end
