
% This code has been adapted from part of the supplementary material to the ICCV 2013 paper
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
% Contact the original author: ttommasi [at] esat.kuleuven.be
%

clear

addpath ./functions/
addpath ./flann/
%select('amazon');
%select('caltech');

params.trainingSamples = 20;
spath = '~/data/desc/office/amazon/all_32_3_extra_hybrid_mean/';
tpath = '~/data/desc/caltech10/all_32_3_extra_hybrid_mean/';
categories = {'backpack.hdf5' 'headphones.hdf5' 'monitor.hdf5' 'bike.hdf5' 'keyboard.hdf5' 'mouse.hdf5' 'projector.hdf5' 'calculator.hdf5'  'laptop.hdf5' 'mug.hdf5'};
s = getImageIDs(spath, categories);
t = getImageIDs(tpath, categories);
S.indexes = s;
S.path = spath;
T.indexes = t;
T.path = tpath;
params.SourceDataset = S;
params.TargetDataset = T;

[accuracyIN]=run_UnsupervisedNBNN(params);
fprintf('\nNBNN Caltech10->Caltech10, rec. rate: %.2f %%', accuracyIN);
fprintf('\n');
