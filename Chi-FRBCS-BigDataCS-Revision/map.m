function map(data, mfCenters, mfSegments, minMisclassificationCost, majMisclassificationCost,...
             majClassLable, minClassLable, intermKVStore)
    tic
    dataArray = table2array(data);    
    % RB(if-part(10), then-part, RW)
    N = size(dataArray, 1);
    d = size(dataArray, 2) - 1;
    M = size(mfCenters, 2);
    maxPartiotion = zeros(N, d, 2); 
    matchingDegrees = zeros(N, d, M);%membership degree for every feature in every mf 
    RB = zeros(N, d + 2);
    rulesMatchingDegrees = ones(N, N); %degree of every sample in every rule
    r = 1;
    
    %% Calculating matching degrees
    for i = 1 : N
        for h = 1 : d
            for m = 1 : M
                if m == 1
                    left = mfCenters(h, m) - 1;
                else
                    left = mfCenters(h, m - 1);
                end
                if m == 5
                    right = mfCenters(h, m) + 1;
                else
                    right = mfCenters(h, m + 1);
                end
                matchingDegrees(i, h, m) = trimf(dataArray(i, h), [left, mfCenters(h, m), right]);
            end
            if h == 2 || h == 3 || h == 4
                maxPartiotion(i, h, 2) = dataArray(i, h);
                maxPartiotion(i, h, 1) = 1;
            else
                maxPartiotion(i, h, 2) = find(mfSegments(h, :) <= dataArray(i, h), 1, 'last');
                maxPartiotion(i, h, 1) = matchingDegrees(i, h, maxPartiotion(i, h, 2));
            end
        end
    end
    
    %% Creating rules
    for i = 1 : N        
        antecedent = zeros(1, d);
        consequent = 0;
     
        for k = 1 : d
            antecedent(k) = maxPartiotion(i, k, 2);
        end
        consequent = dataArray(i, end);
        if r == 1
            RB(r, 1:d) = antecedent;
            RB(r, d + 1) = consequent;
            [RB(r, d + 2), rulesMatchingDegrees(r, :)] = ruleWeightComputer([RB(r, 1:d), RB(r, d + 1), 0], -1, matchingDegrees,...
                dataArray, mfCenters, minMisclassificationCost, majMisclassificationCost, majClassLable, minClassLable);           
            r = r + 1;
        else
            duplicates = 0;
            for u = 1 : r - 1
                if isequal(RB(u, 1:d), antecedent)
                    duplicates = u;
                    break;
                end
            end
            if (duplicates > 0)                
                if RB(duplicates, d + 1) ~= consequent
                    [w, ~] = ruleWeightComputer([antecedent, consequent, 0], rulesMatchingDegrees(duplicates, :), matchingDegrees,...
                        dataArray, mfCenters, minMisclassificationCost, majMisclassificationCost, majClassLable, minClassLable);
                    
                    if w > RB(duplicates, d + 2)
                        RB(duplicates, d + 2) = w;
                        RB(duplicates, d + 1) = consequent;
                    end
                end
            else
                RB(r, 1:d) = antecedent;
                RB(r, d + 1) = consequent;
                [RB(r, d + 2), rulesMatchingDegrees(r, :)] = ruleWeightComputer([RB(r, 1:d), RB(r, d + 1), 0], -1, matchingDegrees, dataArray,...
                    mfCenters, minMisclassificationCost, majMisclassificationCost, majClassLable, minClassLable);
                r = r + 1;
            end
        end 
    end
    for y = 1 : r - 1
        add(intermKVStore, mat2str(RB(y, 1:d)), RB(y, :));
    end
    toc
end
