function doMOP(filelistpath, nLevels, outpath)
    FSIZE=4096
    filelist = dataread('file', filelistpath, '%s', 'delimiter', '\n');
    [savedFilenames, datasetNames] = savePooledFeatures(filelist, nLevels, FSIZE);
    saveConcatenatedFeatures(savedFilenames, outpath, ...
                             datasetNames);
    disp('All done!')
end

function [savedFilenames, datasetNames] = savePooledFeatures(filelist, nLevels, FSIZE)
%    keyboard
    [savedName, datasetNames] = saveGlobalFeatures(filelist, FSIZE);
    savedFilenames = { savedName };
    for l=1:nLevels
        [D, V, PCAV] = newLearnCodebook(filelist, 500, 100, l);
        X = newBuildVLADALL(filelist, D, V, PCAV, l);
        savename = strcat('tmp_', num2str(l));
        save( savename, 'X','D','V','PCAV','-v7.3');
        savedFilenames{end+1} = savename;
    end
end

function saveConcatenatedFeatures(savedFilenames, outpath, datasetNames)
	 X = [];
	 for x=1:numel(savedFilenames)
             data = load(savedFilenames{x});
	     X = [X data.X];
	 end
	 save(outpath, 'X', 'datasetNames', '-v7.3');
end

function [savename, datasetNames] = saveGlobalFeatures(img_list, FSIZE)
    n_images = length(img_list);
    X = zeros(n_images, FSIZE, 'single');
    datasetNames = cell(n_images, 1);
    for i=1:n_images
        % load each data separately
        name = img_list{i};
        if mod(i,500)==0
            fprintf('Loading image %s, %d/%d\n',name, i, n_images);
        end
        try
            v = h5read(name, '/feats', [1, 1], [FSIZE, 1]);  % read only the first element
            datasetNames{i} = h5readatt(name, '/', 'relative_path');
        catch ME
            disp('wrong image')
            v = randn(FSIZE, 1);
            v(v<0)=0;
        end
        X(i,:) = v(:)';
    end
    savename = 'tmp_whole';
    save( savename, 'X','-v7.3');
end
