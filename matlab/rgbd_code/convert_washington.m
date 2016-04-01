filelist = importdata('all_depth.txt');
source_dir = '/mnt/BackupB/data/images/rgbd-dataset';
output_dir = '/mnt/BackupB/data/images/rgbd-dataset-8bit';
for i=1:numel(filelist)
    in = fullfile(source_dir, filelist{i});
    out = fullfile(output_dir, filelist{i});
    convert_16to8(in,out);
end