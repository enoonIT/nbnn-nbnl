function doMOP(filelist, nLevels, outpath)
    FSIZE=4096
    savedFilenames = savePooledFeatures(filelist, nLevels, FSIZE);
    saveConcatenatedFeatures(savedFilenames);
end

function savedFilenames = savePooledFeatures(filelist, nLevels, FSIZE)
    keyboard
    savedFilenames = { saveGlobalFeatures(filelist, FSIZE) };
    for l=1:nLevels
        [D, V, PCAV] = newLearnCodebook(filelist, 500, 100, l);
        X = newBuildVLADALL(filelist, D, V, PCAV, l);
        savename = strcat('tmp_', l);
        save( savename, 'X','D','V','PCAV','-v7.3');
    end
end

function saveConcatenatedFeatures(savedFilenames, outpath)
	 X = []
	 for x=1:numel(savedFilenames)
	     X = [X load(savedFilenames{x})];
	 end
	 save(outpath, 'X', '-v7.3');
end

function savename = saveGlobalFeatures(img_list, FSIZE)
    n_images = length(img_list);
    X = zeros(n_images, FSIZE, 'single');
    for i=1:n_images
        % load each data separately
        name = img_list{i};
        fprintf('Loading image %s, %d/%d\n',name, i, n_images);
        try
            v = h5read(name, '/feats', [1, 1], [FSIZE, 1]);  % read only the first element
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
