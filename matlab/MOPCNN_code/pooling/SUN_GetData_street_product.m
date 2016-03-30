function X = SUN_GetData(patchsize)
    image_root = '/home/fmcarlucci/data/images/street2shop/test_train_splits/';
    
    descRoot = '/home/fmcarlucci/data/desc/';
    descPath = {'product_train','street_train','product_test', 'street_test'};
    dataset_filenames = {'train_product','train_street','test_product', 'test_street'};
    imagePath = {'train/product','train/street','test/product', 'test/street'};
    dataset_mat_path = '/home/fmcarlucci/nbnn-nbnl/matlab/street_product/';
    for ndata = 1:length(dataset_filenames)
        out = strcat(dataset_mat_path, strrep(dataset_filenames{ndata},'/','_'),'.mat');
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
            filenames{a}=strrep(filenames{a}, strcat(image_root, imagePath{ndata}), '');
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
