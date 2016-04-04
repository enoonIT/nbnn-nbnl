
input_folder = '/home/enoon/data/tests/folding_chair';
output_folder = '/home/enoon/data/tests/folding_chair_1';
input_file = '/home/enoon/data/tests/files.txt';
inpt = importdata(input_file);
tic
add_white_noise(inpt, input_folder, output_folder);
toc