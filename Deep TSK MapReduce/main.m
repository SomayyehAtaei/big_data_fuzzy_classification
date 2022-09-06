close all;
clear all;
clc

%% Breast (BRE)
% BRE = load('..\Datasets\Breast (BRE)\breast-cancer-wisconsin.data.txt');
% X = BRE(:, 2:end-1)';
% Y = BRE(:, end)';
% max = max(X')';
% min = min(X')';
% for i = 1 : size(X, 2)
%     X(:, i) = (X(:, i) - min) ./ (max - min);
% end
% majClassLable = 2;
% minClassLable = 4;

%% MIMIC-III Dataset
dataset = importdata('/home/deep/Ataei/Implementation/Datasets/mimic-iii/MICU_65.mat');
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

DP = 3;
tic  
for i = 1 : 10
    i
    K = round(6 + 9 * rand(DP, 1)); % Number of fuzzy rules are selected in [3, 9] normally
    
%     [accuracy(i, 1), TP(i, 1), TN(i, 1), FP(i, 1), FN(i, 1), rulesNumber(i, 1)] = ...
%                                 DeepLearning(X_train, Y_train, DP, K);
    
    
    [AUC(i, 1), accuracy(i, 1), TP(i, 1), TN(i, 1), FP(i, 1), FN(i, 1), rulesNumber(i, 1)] = ...
                                DeepLearningWeightening(X_train, Y_train, DP, K, majClassLable, ...
                                majClassLable, majMisclassificationCost, minMisclassificationCost);
    [AUC_test(i, 1), accuracy_test(i, 1), TP_test(i, 1), TN_test(i, 1), FP_test(i, 1), FN_test(i, 1), rulesNumber_test(i, 1)] = Prediction(X_test, Y_test, DP, K); 
end
toc

table(AUC, accuracy, TP, TN, FP, FN, AUC_test, accuracy_test, TP_test, TN_test, FP_test, FN_test, rulesNumber)
mean(AUC)
mean(AUC_test)