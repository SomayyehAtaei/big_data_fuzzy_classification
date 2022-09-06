function rules = clusteringRules(finalRules, majClassLable, minClassLable)
    NC = 150;
    NR = size(finalRules, 1);
    d = size(finalRules, 2) - 2;
    clusteredFinalRules = zeros(NR, 1);
    clustersConsequent = zeros(NC, 1);
    rules = zeros(NC, d + 2);

    clusteredFinalRules = kmeans(finalRules(:, 1:d), NC, 'Distance', 'hamming');
        
    for c = 1 : NC
        minWeight = 0;
        majWeight = 0;
        check = true;
        
        for r = 1 : NR
            if clusteredFinalRules(r) == c
                if check == true
                    rules(c, 1:d) = finalRules(r, 1:d);
                    check = false;
                end
                if finalRules(r, d + 1) == majClassLable
                    majWeight = majWeight + finalRules(r, d + 2);
                else
                    minWeight = minWeight + finalRules(r, d + 2);
                end
            end
        end
        if majWeight > minWeight
            rules(c, d + 1) = majClassLable;
            rules(c, d + 2) = majWeight;
        else
            rules(c, d + 1) = minClassLable;
            rules(c, d + 2) = minWeight;
        end
    end
end

