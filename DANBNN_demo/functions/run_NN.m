
% This code is part of the supplementary material to the ICCV 2013 paper
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


function [accuracyIN, accuracyCROSS]=run_NN(source, target)

sset=load(['./splits/' source '_select.mat']);
tset=load(['./splits/' target '_select.mat']);
position=['./Office+Caltech/amazon_vocabulary_BOW/'];

% LOAD TARGET DATA
name=[ target '_hist.mat'];
T=load([position name]); 
clear name
tt = T.label;
uy=unique(T.label);
ny=numel(uy);
pp = T.fts;
pp = pp ./ repmat(sum(pp,2),1,size(pp,2)); 
pp = zscore(pp,1);  
clear T

% LOAD SOURCE DATA
name=[source '_hist.mat'];
S=load([position name]); 
clear name
t=S.label;
p = S.fts;
p = p ./ repmat(sum(p,2),1,size(p,2)); 
p = zscore(p,1);  
clear S

str=[];
sytr=[];
tte=[];
tyte=[];
ttr=[];
tytr=[];

for c=1:ny

    % source training
    str_ind=sset.id_tr{c};
    str=[str;p(str_ind,:)];
    sytr=[sytr t(str_ind)];
    clear str_ind

    % target training
    ttr_ind=tset.id_tr{c};
    ttr=[ttr;pp(ttr_ind,:)];
    tytr=[tytr tt(ttr_ind)];
    clear ttr_ind 
    
    % target test
    tte_ind=tset.id_te{c};
    tte=[tte;pp(tte_ind,:)];
    tyte=[tyte tt(tte_ind)];
    clear tte_ind
end

[idx, ~]=kNearestNeighbors(str, tte, 1);
predict_labels=sytr(idx);
accuracyCROSS=sum(predict_labels==tyte)/numel(tyte)*100;
clear idx predict_labels

[idx, ~]=kNearestNeighbors(ttr, tte, 1);
predict_labels=tytr(idx);
accuracyIN=sum(predict_labels==tyte)/numel(tyte)*100;
clear idx predict_labels

end