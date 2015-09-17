function doDANBNN(data_folder, jobId)
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

    myAddPath ./functions/
    myAddPath ./flann/
    output_folder = strcat(data_folder,'/outputDANBNN/');
    if ~exist(output_folder, 'dir')
      mkdir(output_folder);
    end
    outName = strcat(output_folder,'job_DANBNN_Relu_',num2str(jobId),'.mat')
    if exist(outName,'file')
        disp('Job already performed - skipping');
        return
    end
    % spath = '~/data/desc/office/amazon/all_32_3_extra_hybrid_mean/';
    % tpath = '~/data/desc/caltech10/all_32_3_extra_hybrid_mean/';
    categories = {'backpack.hdf5' 'headphones.hdf5' 'monitor.hdf5' 'bike.hdf5' 'keyboard.hdf5' 'mouse.hdf5' 'projector.hdf5' 'calculator.hdf5'  'laptop.hdf5' 'mug.hdf5'};
    params = gridJobInterpreter(jobId,data_folder, categories);
    splits = 1;
    accuracy = zeros(splits,1);
    for i=1:splits
        fprintf('\nSplit %d:\n',i);
        [accuracyIN]=run_UnsupervisedDANBNN(params);
        fprintf('\nDANBNN %s->%s, rec. rate: %.2f %%\n', params.SourceDataset.dataset, params.TargetDataset.dataset, accuracyIN);
        accuracy(i) = accuracyIN;
    end
    save(outName,'params','accuracy');
end
