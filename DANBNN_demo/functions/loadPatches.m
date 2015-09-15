function selectedPatches = loadPatches(imageIds, dataset, relu)
    patchIndexes = imageIds(:,2:3);
    patches = h5read(dataset,'/patches');
    patchIndexes(:,1) = patchIndexes(:,1) + 1;
    N = size(patches,2); % magic from stackoverflow to extract the desired patches https://stackoverflow.com/questions/32524204/extract-rows-from-matrix-according-to-range-indexing
    [R,~] = find((bsxfun(@le,patchIndexes(:,1),1:N) & bsxfun(@ge,patchIndexes(:,2),1:N)).');
    selectedPatches = single(patches(:,R));
    if(relu)
        negs = find(selectedPatches<0);
        selectedPatches(negs) = 0;
        fprintf('Applying RELU\n');
    end
%     for x=1:size(patchIndexes,1)
%         patchIndexes(x,1)
%         patchIndexes(x,2)
%         selectedPatches = [selectedPatches patches(:,patchIndexes(x,1):patchIndexes(x,2))]; %first index is inclusive, second is exclusive
%     end
end