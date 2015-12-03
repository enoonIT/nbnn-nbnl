function selectedPatches = loadPatchesDA(imageIds, dataset, relu, addPos, posScale)
%   selectedPatches is featDescriptor x numPatches
    patchIndexes = imageIds(:,2:3);
    patches = h5read(dataset,'/patches7');
    if(addPos)
        positions = h5read(dataset, '/positions');
        patches = [patches; (positions*posScale)];
        fprintf('Adding position to features\n');
    end
    patchIndexes(:,1) = patchIndexes(:,1) + 1;
    N = size(patches,2); % magic from stackoverflow to extract the desired patches https://stackoverflow.com/questions/32524204/extract-rows-from-matrix-according-to-range-indexing
    [R,~] = find((bsxfun(@le,patchIndexes(:,1),1:N) & bsxfun(@ge,patchIndexes(:,2),1:N)).');
    selectedPatches = single(patches(:,R));
    if(relu)
        negs = find(selectedPatches<0);
        selectedPatches(negs) = 0;
        fprintf('Applying RELU\n');
    end
    fprintf('Loaded %d patches\n', size(selectedPatches,2));
end