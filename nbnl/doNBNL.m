% clear
% clc

algM = 100
algP = 1.5
algIter= 30

first_split = 1;
last_split = 5;

% data_folder = '~/data';
% jobId = 1;

output_folder = strcat(data_folder,'/output/');
if ~exist(output_folder, 'dir')
  mkdir(output_folder);
end
[lambda, input_folder] = jobInterpreter(jobId);
dataset_dir = strcat(data_folder, '/desc/scene15/',input_folder, '/relu/')

algo = ML3();
algo.m = algM;
algo.maxCCCPIter = algIter;
algo.p = algP;
algo.lambda = lambda;%2e-2

% containers used to record split specific data
splitAccuracy = zeros(1 + last_split - first_split, 1);
trainingTime = zeros(1 + last_split - first_split, 1);
trainingAccuracy = zeros(1 + last_split - first_split, 1);
testingTime = zeros(1 + last_split - first_split, 1);
testingAccuracy = zeros(1 + last_split - first_split, 1);

for split = first_split : last_split    
    % get training patches
    [training_features, training_labels, ~, scalingFactor, featureMean, Ured] = preAllocGetPatches(dataset_dir, split, 'train');
    % train ML3 model
    tic, model=algo.train(training_features, training_labels); trainingTime(split) = toc
%     save(modelName,'model','algo')
    % test training features
    [~,~,accuracy,confusion]=algo.test(training_features,training_labels,model);
    disp('Training accuracy');
    trainingAccuracy(split) = accuracy;
    confusion
    clear training_features;
    %load testing features
    [test_features, test_labels, trainingDataInfo] = preAllocGetPatches(dataset_dir, split, 'test', scalingFactor, featureMean, Ured);
    disp 'Testing'
    tic, [classScores,predict_labels,accuracy,confusion]=algo.test(test_features,test_labels,model); testingTime(split) = toc
    confusion
    testingAccuracy(split) = accuracy;
    
    % Apply NBNL
    [ confusion, accuracy ] = NBNL( classScores, trainingDataInfo )
    splitAccuracy(split) = accuracy;
    
    clear test_features; %to avoid memory problems
end
splitAccuracy
fprintf('Mean training time: %f hours\nMean training accuracy: %f\n',mean(trainingTime)/3600, mean(trainingAccuracy));
fprintf('Mean testing time %f\nMean testing accuracy %f\n',mean(testingTime), mean(testingAccuracy));
outName = strcat(output_folder,'job_',num2str(jobId));
save(outName,'lambda','input_folder','splitAccuracy','trainingAccuracy','testingAccuracy','trainingTime','testingTime');