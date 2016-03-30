clear
descRoot = '/home/fmcarlucci/data/desc/';
datasets = {'art/bycateg/test','art/bycateg/train','art/bycateg/validation', 'VOC/TrainVal/VOCdevkit/VOC2011/bycategory'};
descPath = {'ART_test','ART_train', 'ART_validation', 'VOC'};
imagePath = {'art/bycateg/test','art/bycateg/train', 'art/bycateg/validation', 'VOC/TrainVal/VOCdevkit/VOC2011/bycategory'};
dataset_mat_path = '/home/fmcarlucci/nbnn-nbnl/matlab/art_voc/data/';
dataset_filenames_path = '/home/fmcarlucci/nbnn-nbnl/matlab/art_voc/';
elements=1:numel(descPath);


for idx=1
    target_dataset = descPath{idx};
	fprintf('Start cycle for target %s\n', target_dataset);

    T128 = load(strcat(dataset_mat_path,target_dataset,'_128.mat'));
    T064 = load(strcat(dataset_mat_path,target_dataset,'_64.mat'));
    
	for dest=elements
        source_dataset_name = descPath{dest};
        short_name = source_dataset_name;
        matfilename = strcat(dataset_mat_path, target_dataset,'_to_',short_name,'_VLAD.mat');
        if exist(matfilename,'file')
            fprintf('%s already exists, skipping\n', matfilename);
            continue;
        end
        data_filenames = strcat(dataset_filenames_path, strrep(datasets{dest},'/','_'),'.mat');
        load(data_filenames);
        for a=1:numel(filenames) %clean the filenames
            filenames{a}=strrep(filenames{a}, strcat('/home/fmcarlucci/data/images/', imagePath{dest}), '');
        end
        
		fprintf('Applying to %s\n', short_name)
        S256 = load(strcat(dataset_mat_path,short_name,'_256.mat'));
        data = S256.X;
        %128px wide patches
        X128 = buildVLADALL(strcat(descRoot, source_dataset_name, '/patch_grid_128'), ...
                filenames, T128.D, T128.V, T128.PCAV);
        X64  = buildVLADALL(strcat(descRoot, source_dataset_name, '/patch_grid_64'), ...
                filenames, T064.D, T064.V, T064.PCAV);
        data = [data, X128, X64];
		save(matfilename, 'data');
    end
end