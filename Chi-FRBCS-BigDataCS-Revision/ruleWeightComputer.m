function [RWi, rulesMatchingDegreeArray]= ruleWeightComputer(RBi, rulesMatchingDegree, matchingDegrees, ...
                                          dataArray, mfCenters, minMisclassificationCost, ...
                                          majMisclassificationCost, majClassLable, minClassLable)
    N = size(dataArray, 1);
    d = size(dataArray, 2) - 1;
    CjMu = 0;
    NCjMu = 0;
    p = RBi(1:d);
    
    if rulesMatchingDegree == -1
        rulesMatchingDegreeArray = ones(N, 1);

        for j = 1 : N
            temp = p;
     
            for k = 1 : d
                rulesMatchingDegreeArray(j) = rulesMatchingDegreeArray(j) * matchingDegrees(j, k, temp(k));
            end
        end
    else
        rulesMatchingDegreeArray = rulesMatchingDegree;
    end
    
    for j = 1 : N
        if dataArray(j, end) == RBi(1, d + 1)
            if dataArray(j, end) == majClassLable
                CjMu = CjMu + rulesMatchingDegreeArray(j) * majMisclassificationCost;
            else
                CjMu = CjMu + rulesMatchingDegreeArray(j) * minMisclassificationCost;
            end                
        else
            if dataArray(j, end) == majClassLable
                NCjMu = NCjMu + rulesMatchingDegreeArray(j) * majMisclassificationCost;
            else
                NCjMu = NCjMu + rulesMatchingDegreeArray(j) * minMisclassificationCost;
            end                
        end
    end
    RWi = 0;
    if (CjMu + NCjMu) ~= 0
        RWi = (CjMu - NCjMu) / (CjMu + NCjMu);
    end
end

