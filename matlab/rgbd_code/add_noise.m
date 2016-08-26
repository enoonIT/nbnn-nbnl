
input_folder = '/mnt/SSD/DepthNet/Renderings/8bit/';
output_folder = '/mnt/SSD/DepthNet/Renderings/8bits_noisy/';
input_file = '/mnt/SSD/DepthNet/Renderings/files.txt';
inpt = importdata(input_file);
tic
add_white_noise(inpt, input_folder, output_folder);
toc