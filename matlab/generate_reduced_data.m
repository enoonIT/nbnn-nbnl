clear
LEVEL=7;
patch=strcat('/patches',num2str(LEVEL));
datasets{1}.data = load_patches('~/data/desc/whole_image/caltech10/', patch);
datasets{1}.name = 'caltech';
datasets{2}.data = load_patches('~/data/desc//whole_image/office/amazon10/', patch);
datasets{2}.name = 'amazon';
datasets{3}.data = load_patches('~/data/desc/whole_image/office/dslr10/', patch);
datasets{3}.name = 'dslr';
datasets{4}.data = load_patches('~/data/desc/whole_image/office/webcam10/', patch);
datasets{4}.name = 'webcam';
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
for idx=elements
	name = datasets{idx}.name;
	fprintf('Start cycle for target %s\n', name)
	X = cell_to_matrix(datasets{idx}.data);
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
end
