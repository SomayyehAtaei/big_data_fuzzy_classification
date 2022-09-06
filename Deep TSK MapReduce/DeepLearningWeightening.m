function [AUC, accuracy, TP, TN, FP, FN, rulesNumber] = DeepLearningWeightening(X, Y_actual, DP, K, majClassLable, ...
                                minClassLabel, majMisclassificationCost, minMisclassificationCost)

    gpuArray(X);
    gpuArray(Y_actual);
    d = size(X, 1);
    N = size(X, 2);
    numberOfPartitions = 7;% max(G)
    epsilon = 1e-20;
    IterMax = 40;
%     etha = 0.007;
    etha = 0.15;
    clear temp mu_y1 mu_y2 theta omega G sigma_G centers q_g
    q_g = -1 + 2 * rand(DP, max(K), DP); % layer-rule-coefficient    
    theta = zeros(DP, d + DP, numberOfPartitions, max(K)); 
    omega = zeros(DP, d + DP, max(K));
    G = zeros(DP, 1);
    sigma_G = zeros(DP, numberOfPartitions);
    centers = zeros(DP, numberOfPartitions);
    accuracy = 0;
    majClassLable = 1;
    minClassLabel = -1;
    W = ones(DP, max(K));
    
    for dp = 1 : DP   
        switch dp
            case 2
                X = [X; y1'];
                clear temp mu_y1 mu_y2;
            case 3
                X = [X; y2'];
                clear temp mu_y1 mu_y2;
        end
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
        sigma_G(dp, 1:G(dp)) = 0.5 * (1 / G(dp)) + 0.5 * (1 / G(dp)) * rand(G(dp), 1);
        
        %% Determine centers of gaussian membership functions in current depth in an uniform partitioning
        centers(dp, 1:G(dp)) = 0 : (1 / (G(dp) - 1)) : 1;

        %% Randomize rule combination matrix in current depth        
        % if-part of each rule in layer dp is consist of d diemention of input and (dp-1) output of prev layers.                
        for k = 1 : K(dp)
            for j = 1 : (d + dp - 1)
                temp2 = floor(1 + (G(dp) - 0.01) * rand);
                theta(dp, j, temp2, k) = 1;
            end
        end

        %% Randomize feature selection matrix in current depth                
        for k = 1 : K(dp)
            for j = 1 : d
                temp2 = rand;
                if temp2 > 0.5
                    omega(dp, j, k) = 1;
                end
            end
            if dp > 1
                for z = 1 : dp - 1
                    omega(dp, d + z, k) = 1;
                end
            end
        end

        %% Map features to guassian kernel space
        U = zeros(G(dp), d + (dp - 1), N);
        for i = 1 : N
            for j = 1 : d + (dp - 1)
                minBound = min(X(j, :));
                maxBound = max(X(j, :));
                for m = 1 : G(dp)
%                     U(m, j, i) = exp((-(X(j, i) - (minBound + (maxBound - minBound) * centers(dp, m))) ^ 2) / (2 * (sigma_G(dp, m) ^ 2)));
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
%                                         V(k, j, i) = 1 - prod(1 - squeeze(theta(dp, j, 1:G(dp), k))' * U(:, j, i));
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
%                 H(k, i) = prod(V(k, (1:d), i));
                H(k, i) = min(V(k, (1:d), i));
            end
        end
%         H = log(H);

        %% Learn the parameters in the then-parts in fuzzy rules of the current layer dp using GD
        t = 1;
        E_current = 1e20;
        E_prev = 1e21;        
        y_rule = zeros(DP, N, max(K));
        switch dp
            case 1
%                 while ((((E_prev - E_current) > epsilon) || ((E_current - E_prev) < 1e15)) && (t < IterMax))% || t < 3
                while (((E_prev - E_current) > epsilon) && (t < IterMax))% || t < 3
                    A1 = zeros(N, 1);                         
                    for i = 1 : N
                        y_rule(1, i, 1:K(dp)) = q_g(1, 1:K(dp), 1) .* H(:, i)';
                        A1(i, 1) = q_g(1, 1:K(dp), 1) * (H(:, i) .* W(dp, 1:K(dp))');
                    end
                    y1 = A1;
                    y_predicted = squeeze(y_rule(1, :, 1:K(dp)));
                    W = RuleWeightComputer(X, H, K, W, y_predicted, Y_actual, dp, majClassLable, ...
                                minClassLabel, majMisclassificationCost, minMisclassificationCost);
                    E_prev = E_current;
                    E_cost = 0.1 * ones(size(Y_actual));
                    for i = 1 : N
                        if Y_actual(1, i) == minClassLabel && y1(i) > 0 % minarity class label always is negative class
                            E_cost(1, i) = minMisclassificationCost / 10;
                        end
                    end
                    Diff = (Y_actual - y1') .* (Y_actual - y1') .* E_cost;                    
%                     FN = 0;
%                     FP = 0;
%                     for i = 1 : N
%                         if Y_actual(i) ~= y1(i)
%                             if Y_actual(i) == 1
%                                 FN = FN + 1;
%                             elseif Y_actual(i) == -1
%                                 FP = FP + 1;
%                             end
%                         end
%                     end
                    E_current = (1/2) * (sum(Diff));% + 0.1 * (FP * minMisclassificationCost / 10 + FN));
                    % update the parameters in then-part of first layer
                    q_g(1, 1:K(dp), 1:dp) = q_g(1, 1:K(dp), 1:dp) - etha * ((-1 * H) * (Y_actual - y1')')';
%                      q_g(1, 1:K(dp), 1:dp) = q_g(1, 1:K(dp), 1:dp) - etha * ((-1 * H) * Diff')';
                    t = t + 1;
                end                

                t1 = t
                y_dp = y1;
%                             y1 = (y1 - min(y1)) ./ (max(y1) - min(y1));
                for i = 1 : N
                    if y1(i) > 0 
                        y1(i) = 1;
                    else
                        y1(i) = 0;
                    end
                end          
            case 2
%                 etha = 0.007;
                E_current = 1e20;
                E_prev = 1e21;
                for i = 1 : N
                    for k = 1 : K(dp)
%                         mu_y1(k, i) = squeeze(theta(dp, j, 1:G(dp), k))' * U(:, d + 1, i);
mu_y1(k, i) = min(V(k, d+1, i));
                    end
                end
                while (((E_prev - E_current) > epsilon) && (t < IterMax))% || t < 3
                    A2 = zeros(N, 1);

                    for i = 1 : N
                        y_rule(1, i, 1:K(dp)) = q_g(1, 1:K(dp), 1) .* H(:, i)';
                        y_rule_B2_1(i, 1:K(dp)) = q_g(2, 1:K(dp), 2) .* (mu_y1(:, i) .* H(:, i))';
                        y_rule(2, i, 1:K(dp)) = (q_g(2, 1:K(dp), 1) .* (mu_y1(:, i) .* H(:, i))') + ...
                                          squeeze(y_rule(1, i, 1:K(dp)))' .* y_rule_B2_1(i, 1:K(dp));
                        A2(i, 1) = (q_g(2, 1:K(dp), 1) .* W(dp, 1:K(dp))) * (mu_y1(:, i) .* H(:, i)); % first coefficients in second layer are r2_k
                        B2_1(i, 1) = (q_g(2, 1:K(dp), 2) .* W(dp, 1:K(dp))) * (mu_y1(:, i) .* H(:, i));
                    end
                    y2 = A2 + A1 .* B2_1;
                    y_predicted = squeeze(y_rule(2, :, 1:K(dp)));
                    W = RuleWeightComputer(X, H, K, W, y_predicted, Y_actual, dp, majClassLable, ...
                                minClassLabel, majMisclassificationCost, minMisclassificationCost);
                    E_prev = E_current;
                    E_cost = 0.1 * ones(size(Y_actual));
                    for i = 1 : N
                        if Y_actual(1, i) == minClassLabel && y1(i) > 0 % minarity class label always is negative class
                            E_cost(1, i) = minMisclassificationCost / 10;
                        end
                    end
                    Diff = (Y_actual - y2') .* (Y_actual - y2') .* E_cost;                    
%                     FN = 0;
%                     FP = 0;
%                     for i = 1 : N
%                         if Y_actual(i) ~= y2(i)
%                             if Y_actual(i) == 1
%                                 FN = FN + 1;
%                             elseif Y_actual(i) == -1
%                                 FP = FP + 1;
%                             end
%                         end
%                     end
                    E_current = (1/2) * (sum(Diff));% + 0.1 * (FP * minMisclassificationCost / 10 + FN));
                    
                    % update the parameters in then-part of second layer
                    q_g(2, 1:K(dp), 1:dp) = q_g(2, 1:K(dp), 1:dp) - etha * ((-1 * H) * (Y_actual - y2')')';
%                      q_g(2, 1:K(dp), 1:dp) = q_g(2, 1:K(dp), 1:dp) - etha * ((-1 * H) * Diff')';
                    t = t + 1;
                    if isnan(y2)
                        disp('22222222222222222222222222222222222222222')
                        mu_y1
                        H(:, i)
                        q_g(2, 1:K(dp), 1)
                        E_current
                        break;
                    end
                end
                t2 = t
                y_dp = y2;
%                     y2 = (y2 - min(y2)) ./ (max(y2) - min(y2));
                for i = 1 : N
                    if y2(i) > 0 
                        y2(i) = 1;
                    else
                        y2(i) = 0;
                    end
                end
            case 3
                etha = 0.008;

                E_current = 1e20;
                E_prev = 1e21;
                for i = 1 : N
                    for k = 1 : K(dp)
%                         mu_y1(k, i) = prod(V(k, d+1, i));
%                         mu_y2(k, i) = prod(V(k, d+1:d+2, i));
                        mu_y1(k, i) = min(V(k, d+1, i));
                        mu_y2(k, i) = min(V(k, d+1:d+2, i));
                    end
                end
                while (((E_prev - E_current) > epsilon) && (t < IterMax))% || t < 3
                    A3 = zeros(N, 1);

                    for i = 1 : N
                        y_rule(1, i, 1:K(dp)) = q_g(1, 1:K(dp), 1) .* H(:, i)';
                        y_rule_B2_1(i, 1:K(dp)) = q_g(2, 1:K(dp), 2) .* (mu_y1(:, i) .* H(:, i))';
                        y_rule(2, i, 1:K(dp)) = (q_g(2, 1:K(dp), 1) .* (mu_y1(:, i) .* H(:, i))') + ...
                                          squeeze(y_rule(1, i, 1:K(dp)))' .* y_rule_B2_1(i, 1:K(dp));
                        
                        y_rule(3, i, 1:K(dp)) = (q_g(3, 1:K(dp), 1)' .* (mu_y2(:, i) .* H(:, i)))' + ...
                            squeeze(y_rule(1, i, 1:K(dp)))' .* (q_g(3, 1:K(dp), 2)' .* (mu_y1(:, i) .* H(:, i)))' + ...
                            squeeze(y_rule(2, i, 1:K(dp)))' .* (q_g(3, 1:K(dp), 3)' .* (mu_y2(:, i) .* H(:, i)))' + ...
                            squeeze(y_rule(1, i, 1:K(dp)))' .* y_rule_B2_1(i, 1:K(dp)) .* ...
                            (q_g(3, 1:K(dp), 3)' .* (mu_y2(:, i) .* H(:, i)))';
                        A3(i, 1) = (q_g(3, 1:K(dp), 1) .* W(dp, 1:K(dp))) * (mu_y2(:, i) .* H(:, i));
                        B3_1(i, 1) = (q_g(3, 1:K(dp), 2) .* W(dp, 1:K(dp))) * (mu_y1(:, i) .* H(:, i));
                        B3_2(i, 1) = (q_g(3, 1:K(dp), 3) .* W(dp, 1:K(dp))) * (mu_y2(:, i) .* H(:, i));
                    end
                    y3 = A3 + A1 .* B3_1 + A2 .* B3_2 + A1 .* B2_1 .* B3_2;
                    y_predicted = squeeze(y_rule(3, :, 1:K(dp)));
                    W = RuleWeightComputer(X, H, K, W, y_predicted, Y_actual, dp, majClassLable, ...
                                minClassLabel, majMisclassificationCost, minMisclassificationCost);
                    E_prev = E_current;
                    
                    E_cost = 0.1 * ones(size(Y_actual));
                    for i = 1 : N
                        if Y_actual(1, i) == minClassLabel && y1(i) > 0 % minarity class label always is negative class
                            E_cost(1, i) = minMisclassificationCost / 10;
                        end
                    end
                    Diff = (Y_actual - y3') .* (Y_actual - y3') .* E_cost;                   
%                     FN = 0;
%                     FP = 0;
%                     for i = 1 : N
%                         if Y_actual(i) ~= y3(i)
%                             if Y_actual(i) == 1
%                                 FN = FN + 1;
%                             elseif Y_actual(i) == -1
%                                 FP = FP + 1;
%                             end
%                         end
%                     end
                    E_current = (1/2) * (sum(Diff));% + 0.1 * (FP * minMisclassificationCost / 10 + FN));
                    
                    q_g(3, 1:K(dp), 1:dp) = q_g(3, 1:K(dp), 1:dp) - etha * ((-1 * H) * (Y_actual - y3')')';
%                     q_g(3, 1:K(dp), 1:dp) = q_g(3, 1:K(dp), 1:dp) - etha * ((-1 * H) * Diff')';
                    t = t + 1;
                    if isnan(y3)
                        disp('33333333333333333333333333333')
                        mu_y1
                        mu_y2
                        H(:, i)
                        q_g(3, 1:K(dp), 1)
                        E_current
                        break;
                    end
                end
                t3 = t
                y_dp = y3; 
                
        end
    end
    save HID-TSK-FC.mat G sigma_G centers theta omega q_g W
    [AUC, accuracy, TP, TN, FP, FN, rulesNumber] = Prediction(X, Y_actual, DP, K); 
end