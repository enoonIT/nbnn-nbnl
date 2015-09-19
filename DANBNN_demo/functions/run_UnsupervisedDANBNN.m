
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

function [accuracyIN]=run_UnsupervisedDANBNN(params)
    tic, [yte te trainData trainIndexes] = getRandomUnsupervisedSplit( params ); toc
    currentTestSample = 1;
    SM.label = [];
    for x=1:numel(trainData) %foreach class
        firstPatch = 1;
        classData = trainData{x}; % descriptor X patches
        size(classData)
        trainId = trainIndexes{x};
        nSamples = size(trainId,1);
        for idx=1:nSamples % assign the test patches to the corresponding test cell
            patchesForSample = trainId(idx,3) - trainId(idx,2); 
            fprintf('From %d to %d\n',firstPatch, (firstPatch+patchesForSample-1));
            SM.feat{currentTestSample} = classData(:, firstPatch:firstPatch + patchesForSample-1);
            firstPatch = firstPatch + patchesForSample;
            currentTestSample = currentTestSample+1;
        end
        SM.label = [SM.label ones(1, nSamples)*x];
    end
    accuracyIN=adaptation_nomem(SM,te,yte);
end
