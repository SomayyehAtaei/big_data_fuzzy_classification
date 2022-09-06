close all;
clear all;
clc

tic
%% KDD Cup Data 10 percent corrected
filename = '..\Implementation\Datasets\KDD\kddcup_10_normal_versus_R2L.txt';
fileID = fopen(filename);
KDD = textscan(fileID, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f', 'Delimiter', {','}, 'EndOfLine', '\n');

fclose(fileID);
X = KDD(:, 1:end-1);
Y = KDD(:, end);
majClassLable = 1;
minClassLable = 2;
d = size(KDD, 2) - 1;

%% Split train and test sets
D1 = load('..\Implementation\Datasets\KDD\kddcup_10_normal_versus_R2L_trainData.txt');
X_train = D1(:, 1:end-1);
Y_train = D1(:, end);

D2 = load('..\Implementation\Datasets\KDD\kddcup_10_normal_versus_R2L_testData.txt');
X_test = D2(:, 1:end-1);
Y_test = D2(:, end);

%% Compute range of each features
range = zeros(size(X_train, 2), 2);
mfCenters = zeros(size(X_train, 2), 5);
mfSegments = zeros(size(X_train, 2), 5); %% Start of segmantation by each mf

for i = 1 : size(X_train, 2)
    range(i, 1) = min(X_train(:, i));
    range(i, 2) = max(X_train(:, i));
    mfCenters(i, 1) = range(i, 1);
    mfCenters(i, 2) = range(i, 1) + (range(i, 2) - range(i, 1)) / 4;
    mfCenters(i, 3) = range(i, 1) + 2 * (range(i, 2) - range(i, 1)) / 4;
    mfCenters(i, 4) = range(i, 1) + 3 * (range(i, 2) - range(i, 1)) / 4;
    mfCenters(i, 5) = range(i, 2);    
            
    mfSegments(i, 1) = range(i, 1);
    mfSegments(i, 2) = range(i, 1) + (range(i, 2) - range(i, 1)) / 8;
    mfSegments(i, 3) = range(i, 1) + 3 * (range(i, 2) - range(i, 1)) / 8;
    mfSegments(i, 4) = range(i, 1) + 5 * (range(i, 2) - range(i, 1)) / 8;
    mfSegments(i, 5) = range(i, 1) + 7 * (range(i, 2) - range(i, 1)) / 8;
end

%% Estimate misclassification cost for each class 
minCount = 0;
majCount = 0;
for i = 1 : size(Y_train, 1)
    if Y_train(i) == majClassLable
        majCount = majCount + 1;
    elseif Y_train(i) == minClassLable
        minCount = minCount + 1;
    end
end
minMisclassificationCost = majCount / minCount;
majMisclassificationCost = 1;

%% Building Knowledge Base MapReduce

ds = datastore('..\Implementation\Datasets\KDD\kddcup_10_normal_versus_R2L_trainData.txt');
ds.ReadSize = 1000;
mapperFunc = @(data, info, interim)map(data, mfCenters, mfSegments, minMisclassificationCost, ...
               majMisclassificationCost, majClassLable, minClassLable, interim);
rules = readall(mapreduce(ds, mapperFunc, @reduce));

save rules
finalRules = zeros(size(rules, 1), d + 2);
for i = 1 : size(rules)
    finalRules(i, :) = cell2mat(rules(i, 2).Value);
end
save finalRules
% load finalRules
% % % finalRules = clusteringRules(finalRules, majClassLable, minClassLable);
%% Classification MapReduce on Train Data
N = size(X_train, 1);
R = size(finalRules, 1);

ds = datastore('..\Implementation\Datasets\KDD\kddcup_10_normal_versus_R2L_trainData.txt');
ds.ReadSize = 1000;
mapperClassifierFunc = @(data, info, interim)map_classifier(data, mfCenters, finalRules, majClassLable, minClassLable, interim);
C = readall(mapreduce(ds, mapperClassifierFunc, @reduce_classifier));

classPredictions = zeros(size(C, 1), 2);
for i = 1 : size(C)
    classPredictions(i, :) = cell2mat(C(i, 2).Value);
end

TP = 0;
TN = 0;
FP = 0;
FN = 0;
AUC = 0;

for i = 1 : N
    if classPredictions(i, 1) == classPredictions(i, 2)
        if classPredictions(i, 1) == majClassLable
            TP = TP + 1;
        elseif classPredictions(i, 1) == minClassLable
            TN = TN + 1;
        end
    else
        if classPredictions(i, 1) == majClassLable
            FN = FN + 1;
        elseif classPredictions(i, 1) == minClassLable
            FP = FP + 1;
        end
    end
end  
accuracy = (TP + TN)/N;
sensitivity = TP / (TP + FN);
specificity = TP / (TN + FP);
rulesNumber = size(finalRules, 1);
AUC = (1 + (TP / (TP + FN)) - (FP / (FP + TN))) / 2;

disp('###################################################################################')
disp('##################     Train Data Classification Results       ####################')
disp('###################################################################################')
table(N, AUC, TP, TN, FP, FN, accuracy, sensitivity, specificity, rulesNumber)

%% Classification MapReduce on Test Data
N = size(X_test, 1);
d = size(X_test, 2);

ds = datastore('..\Implementation\Datasets\KDD\kddcup_10_normal_versus_R2L_testData.txt');
ds.ReadSize = 1000;
mapperClassifierFunc = @(data, info, interim)map_classifier(data, mfCenters, finalRules, majClassLable, minClassLable, interim);
C = readall(mapreduce(ds, mapperClassifierFunc, @reduce_classifier));

classPredictions = zeros(size(C, 1), 2);
for i = 1 : size(C)
    classPredictions(i, :) = cell2mat(C(i, 2).Value);
end

TP = 0;
TN = 0;
FP = 0;
FN = 0;

for i = 1 : size(C)
    if classPredictions(i, 1) == classPredictions(i, 2)
        if classPredictions(i, 1) == majClassLable
            TP = TP + 1;
        elseif classPredictions(i, 1) == minClassLable
            TN = TN + 1;
        end
    else
        if classPredictions(i, 1) == majClassLable
            FN = FN + 1;
        elseif classPredictions(i, 1) == minClassLable
            FP = FP + 1;
        end
    end
end
accuracy = (TP + TN)/N;
sensitivity = TP / (TP + FN);
specificity = TP / (TN + FP);
rulesNumber = size(finalRules, 1);
AUC = (1 + (TP / (TP + FN)) - (FP / (FP + TN))) / 2;
size = N;

disp('###################################################################################')
disp('###################     Test Data Classification Results       ####################')
disp('###################################################################################')
table(N, AUC, TP, TN, FP, FN, accuracy, sensitivity, specificity, rulesNumber)
toc
