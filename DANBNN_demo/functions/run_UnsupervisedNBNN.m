
% This code is adapted from the supplementary material to the ICCV 2013 paper
% "Frustratingly Easy NBNN Domain Adaptation", T. Tommasi, B. Caputo. 
%
% Copyright (C) 2013, Tatiana Tommasi
%
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.
%
% Please cite:
% @inproceedings{TommasiICCV2013,
% author = {Tatiana Tommasi, Barbara Caputo},
% title = {Frustratingly Easy NBNN Domain Adaptation},
% booktitle = {ICCV},
% year = {2013}
% }
%
% Contact the author: ttommasi [at] esat.kuleuven.be
%

function [accuracyIN]=run_UnsupervisedNBNN(params)
    tic, [yte te trainData] = getRandomUnsupervisedSplit( params.SourceDataset, params.TargetDataset, params.trainingSamples, params.relu); toc
    uy=unique(yte)'; %get class labels
    I=eye(size(te{1},1),size(te{1},1)); %identity matrix the size of the feature descriptor (64 for surf, 4096 for CNN)
    tic
    for c=uy     %for each class 
        feat_tr=trainData{c}; %(descriptorSize x nPatches)

        build_params.algorithm='kdtree';
        [index_T, params_T] = flann_build_index(feat_tr, build_params);

        clear tr ntr

        fprintf('Testing NBNN on class %d, with K=%d...\n',c,1);
        K=1;
        for j=1:numel(te)
            feat_te=te{j};
            if(mod(j,50)==0)
                fprintf('Testing sample %d of %d\n', j, numel(te));
            end
            [ii,~]=flann_search(index_T, feat_te, K, params_T); %returns the indexes of the nearest patches for each test sample
            diff=feat_te-feat_tr(:,ii);
            delta=diff';
            dec_values_T(j,c)=trace(delta*I*delta');
            clear delta 
        end
        flann_free_index(index_T);
        clear params_T feate_tr

    end
    [~, predict_labels_T]=min(dec_values_T,[],2);
    size(yte)
    size(predict_labels_T)
    accuracyIN=sum(predict_labels_T==yte)/numel(yte)*100;
    toc
end
