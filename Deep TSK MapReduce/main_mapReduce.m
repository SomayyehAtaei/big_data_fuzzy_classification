close all;
clear all;
clc

tic
%% MIMIC-III Dataset
dataset = importdata('/home/deep/Ataei/Implementation/Datasets/mimic-iii/MICU_65.mat');
% dataset = importdata('E:\Courses\Fuzzy Systems\Final Project\Implementation\Datasets\mimic-iii\MICU_65.mat');
X = dataset.data;
Y = dataset.label';
data = (X - repmat(min(X), size(X,1), 1)) ./ (repmat(max(X),...
                size(X,1), 1) - repmat(min(X), size(X,1), 1));
X = data';
majClassLable = 0;
minClassLable = 1;

%% Split train and test sets
X_train = zeros(size(X, 1), round((2/3) * size(X, 2)));
X_test = zeros(size(X, 1), round((1/3) * size(X, 2)));
Y_train = zeros(1, round((2/3) * size(X, 2)));
Y_test = zeros(1, round((1/3) * size(X, 2)));
majCount = 0;
minCount = 0;
t = 1;
r = 1;
for i = 1 : size(X, 2)
    if rem(i, 3) == 0        
        X_test(:, t) = X(:, i);
        if Y(1, i) == majClassLable
            majCount = majCount + 1;
            Y_test(t) = 1;
        else
            minCount = minCount + 1;
            Y_test(t) = -1;
        end
        t = t + 1;
    else
        X_train(:, r) = X(:, i);
        if Y(1, i) == majClassLable
            majCount = majCount + 1;
            Y_train(r) = 1;
        else
            minCount = minCount + 1;
            Y_train(r) = -1;
        end
        r = r + 1;
    end        
end
minMisclassificationCost = majCount / minCount;
majMisclassificationCost = 1;

% trainData = fopen('/home/deep/Ataei/Implementation/Datasets/mimic-iii/trainData.txt', 'w');
% testData = fopen('/home/deep/Ataei/Implementation/Datasets/mimic-iii/testData.txt', 'w');
% trainData = fopen('E:\Courses\Fuzzy Systems\Final Project\Implementation\Datasets\mimic-iii\MICU_65_trainData.txt', 'w');
% testData = fopen('E:\Courses\Fuzzy Systems\Final Project\Implementation\Datasets\mimic-iii\MICU_65_testData.txt', 'w');
% for i = 1 : size(X_train, 2)
%     str = '';
%     for j = 1 : size(X_train, 1)
%         str = strcat(str, ',', num2str(X_train(j, i)));
%     end
%     str = strcat(str, ',', num2str(Y_train(i)));
%     fprintf(trainData, '%s\n', str);
% end
% for i = 1 : size(X_test, 2)
%     str = '';
%     for j = 1 : size(X_test, 1)
%         str = strcat(str, ',', num2str(X_test(j, i)));
%     end
%     str = strcat(str, ',', num2str(Y_test(i)));
%     fprintf(testData, '%s\n', str);
% end
% fclose('all');

%% Model Creation - Map Reduce
for i = 1 : 3
    tic
    ds = datastore('/home/deep/Ataei/Implementation/Datasets/mimic-iii/MICU_65_trainData.txt');
    ds.ReadSize = 2500;
    numberOfMaps = size(X_train, 2) / ds.ReadSize;

    %% Initializing
    majClassLable = 1;
    minClassLabel = -1;
    DP = 3;
    numberOfPartitions = 7;
    G = zeros(DP, 1);
    sigma_G = zeros(DP, numberOfPartitions);
    centers = zeros(DP, numberOfPartitions);
%     K = round((6 / numberOfMaps) + (9 / numberOfMaps) * rand(DP, 1));
    K = round(30 + 9 * rand(DP, 1));

    for dp = 1 : DP
            %% Randomize number of gaussian membership functions in current depth        
            temp = floor(1 + 3 * rand);
            switch temp
                case 1
                    G(dp) = 3;
                case 2
                    G(dp) = 5;
                case 3
                    G(dp) = 7;
            end

            %% Randomize standard deviation of gaussian membership functions in current depth
            sigma_G(dp, 1:G(dp)) = 0.33 * (1 / G(dp)) + 0.5 * (1 / G(dp)) * rand(G(dp), 1);

            %% Determine centers of gaussian membership functions in current depth in an uniform partitioning
            centers(dp, 1:G(dp)) = 0 : (1 / (G(dp) - 1)) : 1;
    end

    mapperFunc = @(data, info, interim)map_modelCreation(data, DP, K, G, sigma_G, centers, majClassLable,...
                                    minClassLabel, majMisclassificationCost, minMisclassificationCost, interim);
    measures = readall(mapreduce(ds, mapperFunc, @reduce_modelCreation));

    %% Classification - Map Reduce
    ds = datastore('/home/deep/Ataei/Implementation/Datasets/mimic-iii/MICU_65_trainData.txt');
    ds.ReadSize = 2500;
    mapperFunc = @(data, info, interim)map_prediction(data, DP, K, interim);
    measures = readall(mapreduce(ds, mapperFunc, @reduce_prediction));
    results = cell2mat(measures(1, 2).Value);
    AUC(i, 1) = results(1, 1);
    accuracy(i, 1) = results(1, 2);
    TP(i, 1) = results(1, 3);
    TN(i, 1) = results(1, 4);
    FP(i, 1) = results(1, 5);
    FN(i, 1) = results(1, 6);
    rulesNumber(i, 1) = results(1, 7);

    ds = datastore('/home/deep/Ataei/Implementation/Datasets/mimic-iii/MICU_65_testData.txt');
    ds.ReadSize = 2500;
    mapperFunc = @(data, info, interim)map_prediction(data, DP, K, interim);
    measures = readall(mapreduce(ds, mapperFunc, @reduce_prediction));
    results = cell2mat(measures(1, 2).Value);
    AUC_test(i, 1) = results(1, 1);
    accuracy_test(i, 1) = results(1, 2);
    TP_test(i, 1) = results(1, 3);
    TN_test(i, 1) = results(1, 4);
    FP_test(i, 1) = results(1, 5);
    FN_test(i, 1) = results(1, 6);
    toc
end

table(AUC, accuracy, TP, TN, FP, FN, AUC_test, accuracy_test, TP_test, TN_test, FP_test, FN_test, rulesNumber)

AUC = mean(AUC);
accuracy = mean(accuracy);
TP = mean(TP);
TN = mean(TN);
FP = mean(FP);
FN = mean(FN);
AUC_test = mean(AUC_test);
accuracy_test = mean(accuracy_test);
TP_test = mean(TP_test);
TN_test = mean(TN_test);
FP_test = mean(FP_test);
FN_test = mean(FN_test);
rulesNumber = mean(rulesNumber);
table(AUC, accuracy, TP, TN, FP, FN, AUC_test, accuracy_test, TP_test, TN_test, FP_test, FN_test, rulesNumber)
toc









