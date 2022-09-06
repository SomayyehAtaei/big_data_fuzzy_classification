function [W] = RuleWeightComputer(X, H, K, W, y_predicted, Y_actual, dp, majClassLable,...
                                minClassLable, majMisclassificationCost, minMisclassificationCost)

    N = size(X, 2);
    d = size(X, 1);
    CjMu = zeros(K(dp), 1);
    NCjMu = zeros(K(dp), 1);
    
    for j = 1 : N
        for r = 1 : K(dp)
            if y_predicted(j, r) > 0 
                y_predicted(j, r) = 1;
            else
                y_predicted(j, r) = -1;
            end
        end
    end
                
    for j = 1 : N
        for r = 1 : K(dp)
            if Y_actual(j) == y_predicted(1, r)
                if Y_actual(j) == majClassLable
                    CjMu(r) = CjMu(r) + H(r, j) * majMisclassificationCost;
                else
                    CjMu(r) = CjMu(r) + H(r, j) * minMisclassificationCost;
                end                
            else
                if Y_actual(j) == majClassLable
                    NCjMu(r) = NCjMu(r) + H(r, j) * majMisclassificationCost;
                else
                    NCjMu(r) = NCjMu(r) + H(r, j) * minMisclassificationCost;
                end                
            end
        end
    end
    
    for r = 1 : K(dp)
        if (CjMu(r) + NCjMu(r)) ~= 0
            W(dp, r) = (CjMu(r) - NCjMu(r)) / (CjMu(r) + NCjMu(r));
        end
    end
end

