
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

source = params.source;
target = params.target;

% LOAD TARGET DATA
name=['data_' target '.mat'];
T=load(['./Office+Caltech/' name]); %loads features and labels
clear name
uy=unique(T.label); %get class labels

tset=load(['./splits/' target '_select.mat']); %loads the IDs of the testing and testing images id_tr and id_tr

te_ind=[];
yte=[];
for c=uy %for each class
    te_ind=[te_ind tset.id_te{c}]; %loads all testing indexes in one single array
    yte=[yte T.label(tset.id_te{c})]; %same for labels
end
nte=numel(te_ind); %number of test images
te=cell(nte,1); %create a cell for each test image
[te{:}]=deal(T.feat{te_ind}); %loads the features for each image into the corresponding cell

I=eye(size(te{1},1),size(te{1},1)); %identity matrix the size of the feature descriptor (64 for surf, 4096 for CNN)
for c=uy     %for each class 

    % IN-DOMAIN
    ntr=numel(tset.id_tr{c}); %number of training samples for class c
    tr=cell(ntr,1); % create cells for training images
    [tr{:}]=deal(T.feat{tset.id_tr{c}}); %tr now holds the features for the training images of class c
    feat_tr=cell2mat(tr'); % loads them into a single matrix
      
    build_params.algorithm='kdtree';
    [index_T, params_T] = flann_build_index(feat_tr, build_params);
    
    clear tr ntr
    
    fprintf('Testing NBNN on class %d, with K=%d...\n',c,1);
    K=1;
    for j=1:numel(te_ind)
        feat_te=te{j};
        
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
accuracyIN=sum(predict_labels_T==yte')/numel(yte')*100;

end
