function [accuracy, TP, TN, FP, FN, rulesNumber] = DeepLearning(X, Y_actual, DP, K)
    d = size(X, 1);
    N = size(X, 2);
    numberOfPartitions = 7;% max(G)
    epsilon = 1e-5;
    IterMax = 2000;
    etha = 0.06;
    clear temp mu_y1 mu_y2 theta omega G sigma_G centers q_g
    q_g = -1 + 2 * rand(DP, max(K), DP); % layer-rule-coefficient    
    theta = zeros(DP, d + DP, numberOfPartitions, max(K)); 
    omega = zeros(DP, d + DP, max(K));
    G = zeros(DP, 1);
    sigma_G = zeros(DP, numberOfPartitions);
    centers = zeros(DP, numberOfPartitions);
    accuracy = 0;
   
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
                    U(m, j, i) = exp((-(X(j, i) - (minBound + (maxBound - minBound) * centers(dp, m))) ^ 2) / (2 * (sigma_G(dp, m) ^ 2)));
                end
            end
        end

        %% Compute the value of each feature in a fuzzy rule
        V = zeros(K(dp), d + (dp - 1), N);
        for i = 1 : N
            for j = 1 : d + (dp - 1)
                for k = 1 : K(dp)
                    if omega(dp, j, k) == 0
                                        V(k, j, i) = 1 - prod(1 - squeeze(theta(dp, j, 1:G(dp), k))' * U(:, j, i));
%                         V(k, j, i) = 1 - max(1 - squeeze(theta(dp, j, 1:G(dp), k))' * U(:, j, i));
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
                H(k, i) = prod(V(k, (1:d), i));
            end
        end

        %% Learn the parameters in the then-parts in fuzzy rules of the current layer dp using GD
        t = 1;
        E_current = 1;
        E_prev = 0;
        
        switch dp
            case 1
                while (((E_prev - E_current) > epsilon) && (t < IterMax)) || t < 3
                    A1 = zeros(N, 1);                         
                    for i = 1 : N
                        A1(i, 1) = (q_g(1, 1:K(dp), 1) * H(:, i));
                    end
                    y1 = A1;
                    E_prev = E_current;
                    E_current = (1/2) * ((Y_actual - y1') * (Y_actual - y1')');
                    % update the parameters in then-part of first layer
                    q_g(1, 1:K(dp), 1:dp) = q_g(1, 1:K(dp), 1:dp) - etha * ((-1 * H) * (Y_actual - y1')')';
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
%                 A1 = y1;            
            case 2
%                 etha = etha / 10;
                for i = 1 : N
                    for k = 1 : K(dp)
                        mu_y1(k, i) = squeeze(theta(dp, j, 1:G(dp), k))' * U(:, d + 1, i);
                    end
                end
                while (((E_prev - E_current) > epsilon) & (t < IterMax)) || t < 3
                    A2 = zeros(N, 1);

                    for i = 1 : N
                        A2(i, 1) = (q_g(2, 1:K(dp), 1) * (mu_y1(:, i) .* H(:, i))); % first coefficients in second layer are r2_k
                        B2_1(i, 1) = (q_g(2, 1:K(dp), 2) * (mu_y1(:, i) .* H(:, i)));
                    end
                    y2 = A2 + A1 .* B2_1;
                    E_prev = E_current;
                    E_current = (1/2) * sum((Y_actual - y2') * (Y_actual - y2')');
                    % update the parameters in then-part of second layer
                    q_g(2, 1:K(dp), 1:dp) = q_g(2, 1:K(dp), 1:dp) - etha * ((-1 * H) * (Y_actual - y2')')';
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
%                 A2 = y2;
            case 3
%                 etha = etha / 10;
                for i = 1 : N
                    for k = 1 : K(dp)
                        mu_y1(k, i) = prod(V(k, d+1, i));
                        mu_y2(k, i) = prod(V(k, d+1:d+2, i));
                    end
                end
                while (((E_prev - E_current) > epsilon) & (t < IterMax)) || t < 3
                    A3 = zeros(N, 1);

                    for i = 1 : N
                        A3(i, 1) = (q_g(3, 1:K(dp), 1) * (mu_y2(:, i) .* H(:, i)));
                        B3_1(i, 1) = (q_g(3, 1:K(dp), 2) * (mu_y1(:, i) .* H(:, i)));
                        B3_2(i, 1) = (q_g(3, 1:K(dp), 3) * (mu_y2(:, i) .* H(:, i)));
                    end
                    y3 = A3 + A1 .* B3_1 + A2 .* B3_2 + A1 .* B2_1 .* B3_2;
                    E_prev = E_current;
                    E_current = (1/2) * sum((Y_actual - y3') * (Y_actual - y3')');
                    q_g(3, 1:K(dp), 1:dp) = q_g(3, 1:K(dp), 1:dp) - etha * ((-1 * H) * (Y_actual - y3')')';                       
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
    save HID-TSK-FC.mat G sigma_G centers theta omega q_g
    [accuracy, TP, TN, FP, FN, rulesNumber] = Prediction(X, Y_actual, DP, K); 
end