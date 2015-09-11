
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

function select(data)

rand('state',sum(100*clock));

% 20 samples per class in training and all the remaining samples for testing
l=20;

name=['data_' data '.mat'];
S=load(['./Office+Caltech/' name]); 
clear name
uy=unique(S.label);

for c=uy
    cla{c}=find(S.label==c);
end

%for k=1:20   % we considered 20 random splits in the paper
    for c=uy
        r{c}=randperm(numel(cla{c}));
        id_tr{c}=cla{c}(r{c}(1:l));
        id_te{c}=cla{c}(r{c}(l+1:end));        
    end
    %name=[data num2str(k) '_select.mat'];
    name=['./splits/' data '_select.mat'];
    save(name, 'id_tr', 'id_te', 'r');
%end

end
