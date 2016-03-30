clear
datasets{1}.filenames = 'caltechFiles.mat';
datasets{1}.name = 'caltech10';
datasets{2}.filenames = 'amazonFiles.mat';
datasets{2}.name = 'office/amazon10';
datasets{3}.filenames = 'dslrFiles.mat';
datasets{3}.name = 'office/dslr10';
datasets{4}.filenames = 'webcamFiles.mat';
datasets{4}.name = 'office/webcam10';
elements=1:numel(datasets);

dir_name='data/';

    
for idx=elements
	name = datasets{idx}.name;
	fprintf('Start cycle for target %s\n', name)
    %load(datasets{idx}.filenames);
    
    target_short_name = strrep(name, 'office/', '');
%     T256 = load(strcat('data/',saved_name,'_256.mat'));
    T128 = load(strcat('data/',target_short_name,'_128.mat'));
    T064 = load(strcat('data/',target_short_name,'_64.mat'));
    

	for dest=elements
        source_dataset_name = datasets{dest}.name;
        short_name = strrep(source_dataset_name, 'office/', '');
        matfilename = strcat(dir_name, target_short_name,'_to_',short_name,'_VLAD.mat');
        if exist(matfilename,'file')
            fprintf('%s already exists, skipping\n', matfilename);
            continue;
        end
        load(datasets{dest}.filenames);
        for a=1:numel(filenames) %clean the filenames
            filenames{a}=strrep(filenames{a}, strcat('/home/fmcarlucci/data/images/', source_dataset_name), '');
        end
        
		fprintf('Applying to %s\n', short_name)
        S256 = load(strcat('data/',short_name,'_256.mat'));
        data = S256.X;
        %128px wide patches
        X128 = buildVLADALL(strcat('/home/fmcarlucci/data/desc/office_caltech/p128/', source_dataset_name), ...
                filenames, T128.D, T128.V, T128.PCAV);
        X64 = buildVLADALL(strcat('/home/fmcarlucci/data/desc/office_caltech/p64/', source_dataset_name), ...
                filenames, T064.D, T064.V, T064.PCAV);
        data = [data, X128, X64];
		save(matfilename, 'data');
    end
end