
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

function [DD,dec,decte]=fn_create_dist(S,te)

uy=unique(S.label);

for c=uy
    fprintf('Building lookup for class %d\n',c);
    ll=find(S.label==c);
    
    ts=cell(numel(ll),1);
    [ts{:}]=deal(S.feat{ll});
    feat_ALL=cell2mat(ts');
    clear ts
    
    %fprintf('Computing Indexes %d..\n',c);
    build_params.algorithm='kdtree';
    [indexALL, paramsALL(c)] = flann_build_index(feat_ALL, build_params);
    
    K=1;
    for j=1:numel(S.label)

        if (ll~=j) %if different class
            feat_te=S.feat{j};
            [ii,~]=flann_search(indexALL, feat_te, K, paramsALL(c));
            diff=feat_te-feat_ALL(:,ii);
            dec{j,c}=single(diff');
            clear feat_te ii diff
        else %we must remove the image itself from available patches
            lll=ll;
            idx=find(lll==j);
            lll(idx)=[];
            
            ts=cell(numel(lll),1);
            [ts{:}]=deal(S.feat{lll});
            feat_c=cell2mat(ts');
            
            %fprintf('Computing Indexes %d..\n',c);
            build_params.algorithm='kdtree';
            [index, params(c)] = flann_build_index(feat_c, build_params);
            
            feat_te=S.feat{j};
            [ii,~]=flann_search(index, feat_te, K, params(c));
            diff=feat_te-feat_c(:,ii);
            dec{j,c}=single(diff');
            
            clear feat_te feat_c
            flann_free_index(index);
            clear lll idx ts ii diff
        end
        
        DD{j,c}=dec{j,c}'*dec{j,c};
    end
    
    for j=1:numel(te) % compute patch to image class distance for test images
        feat_te=te{j};
        [ii,~]=flann_search(indexALL, feat_te, K, paramsALL(c));
        diff=feat_te-feat_ALL(:,ii);
        decte{j,c}=single(diff');
    end
    flann_free_index(indexALL);
end

end
