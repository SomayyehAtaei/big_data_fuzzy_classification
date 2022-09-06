function map_prediction(data, DP, K, intermKVStore)
    dataArray = table2array(data);
    gpuArray(dataArray);
    X = dataArray(:, 1:end-1)';
    Y_actual = dataArray(:, end)';
    
    load HID-TSK-FC.mat
    d = size(X, 1) - (DP - 1);
    N = size(X, 2);
    
    for dp = 1 : DP
        clear mu_y1 mu_y2
        switch dp
            case 2
                X = [X; y1'];
            case 3
                X = [X; y2'];
        end
        
        %% Map features to guassian kernel space
        U = zeros(G(dp), d + (dp - 1), N);
        for i = 1 : N
            for j = 1 : d + (dp - 1)
                minBound = min(X(j, :));
                maxBound = max(X(j, :));
                for m = 1 : G(dp)
                    U(m, j, i) = exp((-(X(j, i) - centers(dp, m)) ^ 2) / (2 * (sigma_G(dp, m) ^ 2)));
                end
            end
        end
        
        %% Compute the value of each feature in a fuzzy rule
        V = zeros(K(dp), d + (dp - 1), N);
        for i = 1 : N
            for j = 1 : d + (dp - 1)
                for k = 1 : K(dp)
                    if omega(dp, j, k) == 0
                        V(k, j, i) = 1 - min(1 - squeeze(theta(dp, j, 1:G(dp), k))' * U(:, j, i));
                    else
                        V(k, j, i) = 1;
                    end
                end
            end
        end
        
        %% Compute the value of the if-part in a fuzzy rule
        H = zeros(K(dp), N);
        for i = 1 : N
            for k = 1 : K(dp)
                H(k, i) = min(V(k, (1:d), i));
            end
        end
%         H = log(H);
        %%
        switch dp
            case 1
                A1 = zeros(N, 1);                         
                for i = 1 : N
                    A1(i, 1) = q_g(1, 1:K(dp), 1) * (H(:, i) .* W(dp, 1:K(dp))');
                end
                y1 = A1;     
                y_dp = y1;
                for i = 1 : N
                    if y1(i) > 0 
                        y1(i) = 1;
                    else
                        y1(i) = 0;
                    end
                end
            case 2
                for i = 1 : N
                    for k = 1 : K(dp)
                        mu_y1(k, i) = min(V(k, d+1, i));
                    end
                end
                A2 = zeros(N, 1);

                for i = 1 : N
                    A2(i, 1) = (q_g(2, 1:K(dp), 1) .* W(dp, 1:K(dp))) * (mu_y1(:, i) .* H(:, i)); % first coefficients in second layer are r2_k
                    B2_1(i, 1) = (q_g(2, 1:K(dp), 2) .* W(dp, 1:K(dp))) * (mu_y1(:, i) .* H(:, i));
                end
                y2 = A2 + A1 .* B2_1;   
                y_dp = y2;
                for i = 1 : N
                    if y2(i) > 0 
                        y2(i) = 1;
                    else
                        y2(i) = 0;
                    end
                end
            case 3
                for i = 1 : N
                    for k = 1 : K(dp)
                        mu_y1(k, i) = min(V(k, d+1, i));
                        mu_y2(k, i) = min(V(k, d+1:d+2, i));
                    end
                end
                A3 = zeros(N, 1);

                for i = 1 : N
                    A3(i, 1) = (q_g(3, 1:K(dp), 1) .* W(dp, 1:K(dp))) * (mu_y2(:, i) .* H(:, i));
                        B3_1(i, 1) = (q_g(3, 1:K(dp), 2) .* W(dp, 1:K(dp))) * (mu_y1(:, i) .* H(:, i));
                        B3_2(i, 1) = (q_g(3, 1:K(dp), 3) .* W(dp, 1:K(dp))) * (mu_y2(:, i) .* H(:, i));
                end
                y3 = A3 + A1 .* B3_1 + A2 .* B3_2 + A1 .* B2_1 .* B3_2;
                y_dp = y3;
        end           
    end
    TP = 0;
    TN = 0;
    FP = 0;
    FN = 0;
    
    class_prediction = zeros(size(y_dp));
    for i = 1 : N
        if y_dp(i, 1) > 0
            class_prediction(i) = 1;
        elseif y_dp(i, 1) <= 0
            class_prediction(i) = -1;
        end
    end
    for i = 1 : N
        if Y_actual(i) == class_prediction(i)
            if Y_actual(i) == 1
                TP = TP + 1;
            elseif Y_actual(i) == -1
                TN = TN + 1;
            end
        else
            if Y_actual(i) == 1
                FN = FN + 1;
            elseif Y_actual(i) == -1
                FP = FP + 1;
            end
        end
    end
    
%     accuracy = (TP + TN)/N;
%     sensitivity = TP / (TP + FN);    
%     specificity = TP / (TN + FP);
%     rulesNumber = sum(K);
%     TPrate = TP / (TP + FN);
%     FPrate = FP / (FP + TN);
    
%     AUC = (1 + TPrate - FPrate) / 2;
    add(intermKVStore, 1, [TP TN FP FN N sum(K)]);
end


