% clear
% clc

output_folder = strcat(data_folder,'/output/');
if ~exist(output_folder, 'dir')
  mkdir(output_folder);
end
params = gridJobInterpreterDA(jobId, datadir)
outName = strcat(output_folder,'job_NBNL_DA_',num2str(jobId),'.mat')
if exist(outName,'file')
    disp('Job already performed - skipping');
    return
end
algIter = 10
algP = 1.5;
algM = 3;
algo = ML3();
algo.m = algM;
algo.maxCCCPIter = algIter;
algo.p = algP;
% algo.lambda = lambda;%2e-2

    
% get training patches
[training_features, training_labels, testing_features, testing_labels] = prepareData(getRandomSplit(params))
% train ML3 model
tic, model=algo.train(training_features, training_labels); trainingTime = toc
%     save(modelName,'model','algo')
% test training features
[~,~,trainingAccuracy,confusion]=algo.test(training_features,training_labels,model);
clear training_features;
%load testing features
[test_features, test_labels, trainingDataInfo] = preAllocGetPatches(dataset_dir, split, 'test', scalingFactor, featureMean, Ured);
disp 'Testing'
tic, [classScores,predict_labels,accuracy,confusion]=algo.test(test_features,test_labels,model); testingTime = toc
testingAccuracy = accuracy

% Apply NBNL
[ confusion, accuracy ] = NBNL( classScores, trainingDataInfo )
splitAccuracy = accuracy
    

fprintf('Training time: %f hours\nTraining accuracy: %f\n',trainingTime/3600, trainingAccuracy);
fprintf('Testing time %f\nTesting accuracy %f\n',mean(testingTime), testingAccuracy);
save(outName,'algM','dataset_dir','splitAccuracy','trainingAccuracy','testingAccuracy','trainingTime','testingTime','split','algP','confusion');
