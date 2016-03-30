function X = SUN_GetData(patchsize)
    descRoot = '/home/fmcarlucci/data/desc/';
    datasets = {'art/bycateg/test','art/bycateg/train','art/bycateg/validation', 'VOC/TrainVal/VOCdevkit/VOC2011/bycategory'};
    descPath = {'ART_test','ART_train', 'ART_validation', 'VOC'};
    imagePath = {'art/bycateg/test','art/bycateg/train', 'art/bycateg/validation', 'VOC/TrainVal/VOCdevkit/VOC2011/bycategory'};
    dataset_mat_path = '/home/fmcarlucci/nbnn-nbnl/matlab/art_voc/';
    for ndata = 1:length(datasets)
        out = strcat(dataset_mat_path,strrep(datasets{ndata},'/','_'),'.mat');
        load(out);
        dataset=descPath{ndata};
        if(patchsize==256)
            image_dir = strcat(descRoot, dataset, '/whole_image/');
        elseif(patchsize==128)
            image_dir = strcat(descRoot, dataset, '/patch_grid_128/');
        elseif(patchsize==64)
            image_dir = strcat(descRoot, dataset, '/patch_grid_64/');
        end
        for a=1:numel(filenames)
            filenames{a}=strrep(filenames{a}, strcat('/home/fmcarlucci/data/images/', imagePath{ndata}), '');
        end

        savename = dataset;
        if(patchsize==256)
            X = BuildALLGlobal(image_dir, filenames);
            save(strcat(dataset_mat_path,'data/', savename ,'_256'), 'X', '-v7.3')
        elseif(patchsize==128)
            [D, V, PCAV] = learnCodebook(image_dir, filenames, 500, 100);
            X = buildVLADALL(image_dir, filenames, D, V, PCAV);
            save( strcat(dataset_mat_path,'data/',savename ,'_128'),'X','D','V','PCAV','-v7.3')
        elseif(patchsize==64)
            [D, V, PCAV] = learnCodebook(image_dir, filenames, 500, 100);
            X = buildVLADALL(image_dir, filenames, D, V, PCAV);
            save(strcat(dataset_mat_path,'data/',savename ,'_64'),'X','D','V','PCAV', '-v7.3')
        end
    end
end
