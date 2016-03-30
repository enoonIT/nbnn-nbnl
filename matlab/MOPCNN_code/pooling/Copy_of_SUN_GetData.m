function X = SUN_GetData(patchsize)


    load('caltechFiles.mat');
    dataset='caltech10';
    if(patchsize==256)
        image_dir = strcat('/home/fmcarlucci/data/desc/office_caltech/whole_image/', dataset);
    elseif(patchsize==128)
        image_dir = strcat('/home/fmcarlucci/data/desc/office_caltech/p128/', dataset);
    elseif(patchsize==64)
        image_dir = strcat('/home/fmcarlucci/data/desc/office_caltech/p64/', dataset);
    end
    for a=1:numel(filenames)
        filenames{a}=strrep(filenames{a}, strcat('/home/fmcarlucci/data/images/', dataset), '');
    end

    savename = strrep(dataset, 'office/', '');
    if(patchsize==256)
        X = BuildALLGlobal(image_dir, filenames);
        save(strcat('data/',savename ,'_256'), 'X', '-v7.3')
    elseif(patchsize==128)
        [D, V, PCAV] = learnCodebook(image_dir, filenames, 500, 100);
        X = buildVLADALL(image_dir, filenames, D, V, PCAV);
        save( strcat('data/',savename ,'_128'),'X','D','V','PCAV','-v7.3')
    elseif(patchsize==64)
        [D, V, PCAV] = learnCodebook(image_dir, filenames, 500, 100);
        X = buildVLADALL(image_dir, filenames, D, V, PCAV);
        save(strcat('data/',savename ,'_64'),'X','D','V','PCAV', '-v7.3')
    end
end
