function map_classifier(data, mfCenters, finalRules, majClassLable, ...
                        minClassLable, intermKVStore)
    % RB(if-part, then-part, RW)
    N = size(data, 1);
    d = size(data, 2) - 1;
    R = size(finalRules);
    dataArray = table2array(data);
    M = size(mfCenters, 2);
    matchingDegrees = zeros(N, d, M);
    disp('Calculating matching degrees')
    tic
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
        end
    end
toc
disp('Classification')
tic
    %% Classification
    degree = zeros (N, 1);
    C = zeros(N, 1);
    
    for j = 1 : N
        for r = 1 : R
            mul = 1;
            for k = 1 : d
                mul = mul * matchingDegrees(j, k, finalRules(r, k));
            end
            if (mul * finalRules(r, d + 2)) >= degree(j)
                degree(j) = mul * finalRules(r, d + 2);
                C(j) = finalRules(r, d + 1);
            end
        end
    end
  toc
    for i = 1 : N
        add(intermKVStore, '1', [dataArray(i, end) C(i)]);
    end
end