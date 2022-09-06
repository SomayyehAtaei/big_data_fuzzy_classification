function [] = DeepLearning(X, Y_actual, DP, K)
    d = size(X, 1);
    N = size(X, 2);
    epsilon = 1e-5;
    IterMax = 1000;
    etha = 0.008;
    for dp = 1 : DP
        clear temp G_dp sigma_dp c_dp theta_G_dp omega_dp mu_y1 mu_y2
        switch dp
            case 2
                X = [X; (y1 * ones(1, N))];
            case 3
                X = [X; (y1 * ones(1, N)); (y2 * ones(1, N))];
        end
        
        %% Randomize number of gaussian membership functions in current depth        
        temp = round(1 + 2 * rand);
        switch temp
            case 1
                G_dp = 3;
            case 2
                G_dp = 5;
            case 3
                G_dp = 7;
        end

        %% Randomize standard deviation of gaussian membership functions in current depth
        sigma_G_dp = 0.7 + rand(G_dp, 1);

        %% Determine centers of gaussian membership functions in current depth in an uniform partitioning
        c_dp = 0 : (1 / (G_dp - 1)) : 1;

        %% Randomize rule combination matrix in current depth        
        % if-part of each rule in layer dp is consist of d diemention of input and (dp-1) output of prev layers.
        theta_dp = zeros(d + (dp - 1), G_dp, K(dp)); 
        
        for k = 1 : K(dp)
            for j = 1 : d
                for i = 1 : G_dp
                    temp2 = rand;
                    if temp2 > 0.5
                        theta_dp(j, i, k) = 1;
                        break;
                    end
                end
                % for each feature in each rule it's partiotions must determined
                if sum(theta_dp(j, :, k)) == 0
                    temp2 = round(1 + (G_dp - 1) * rand);
                    theta_dp(j, temp2, k) = 1;
                end
            end
            if dp > 1
                for z = 1 : dp - 1
                    temp2 = round(1 + (G_dp - 1) * rand);
                    theta_dp(d + z, temp2, K(dp)) = 1;
                end
            end    
        end
        
        %% Randomize feature selection matrix in current depth        
        omega_dp = zeros(d + (dp - 1), K(dp));
        for k = 1 : K(dp)
            for j = 1 : d
                temp2 = rand;
                if temp2 > 0.5
                    omega_dp(j, k) = 1;
                end
            end
            if dp > 1
                for z = 1 : dp - 1
                    omega_dp(d + z, k) = 1;
                end
            end
        end
        
        %% Map features to guassian kernel space
        U_dp = zeros(G_dp, d + (dp - 1), N);
        for i = 1 : N
            for j = 1 : d + (dp - 1)
                minBound = min(X(j, :));
                maxBound = max(X(j, :));
                for m = 1 : G_dp
                    U_dp(m, j, i) = exp((-(X(j, i) - (minBound + (maxBound - minBound) * c_dp(m))) ^ 2) / (2 * (sigma_G_dp(m) ^ 2)));
                end
            end
        end
        
        %% Compute the value of each feature in a fuzzy rule
        V_dp = zeros(K(dp), d + (dp - 1), N);
        for i = 1 : N
            for j = 1 : d + (dp - 1)
                for k = 1 : K(dp)
                    if omega_dp(j, k) == 0
                        V_dp(k, j, i) = 1 - prod(1 - theta_dp(j, :, k) * U_dp(:, j, i));
                    else
                        V_dp(k, j, i) = 1;
                    end
                end
            end
        end
        
        %% Compute the value of the if-part in a fuzzy rule
        H_dp = zeros(K(dp), N);
        for i = 1 : N
            for k = 1 : K(dp)
                H_dp(k, i) = prod(V_dp(k, (1:d), i));
            end
        end
        
        %% Learn the parameters in the then-parts in fuzzy rules of the current layer dp using GD
            %% Initialize
            t = 1;
            q_g = -1 + 2 * rand(dp, K(dp), dp); % layer-rule-coefficient           
            E_current = 1;
            E_prev = 0;
            
            %% Gradient Descent 
            switch dp
                case 1
                    t = 1;
                    while (E_current - E_prev > epsilon) & (t < IterMax)
                        A1 = zeros(N, 1);                         
                        for i = 1 : N
                            A1(i, 1) = sum(q_g(1, :, 1) * H_dp(:, i));
                        end
                        y1 = A1;                        
                        E_prev = E_current;
                        E_current = (1/2) * sum((Y_actual - y1)' * (Y_actual - y1));
                        % update the parameters in then-part of first layer
                        q_g(1, :, :) = q_g(1, :, :) - etha * ((-1 * H_dp) * (Y_actual' - y1))';
                        t = t + 1;
                    end
                    y_dp = y1;
                case 2
                    for i = 1 : N
                        for k = 1 : K(dp)
                            mu_y1(k, i) = prod(V_dp(k, d+1, i));
                        end
                    end                    
                    t = 1;
                    while (E_current - E_prev > epsilon) & (t < IterMax)
                        A1 = zeros(N, 1);
                        A2 = zeros(N, 1);
                        
                        for i = 1 : N
                            A1(i, 1) = sum(q_g(1, :, 1) * H_dp(:, i)); % in q_g the first coefficient is rL_k
                            A2(i, 1) = sum(q_g(2, :, 1) * mu_y1(:, i) * H_dp(:, i)); % first coefficients in second layer are r2_k
                            B2_1(i, 1) = sum(q_g(2, :, 2) * mu_y1(:, i) * H_dp(:, i));
                        end
                        y2 = A2 + A1 .* B2_1;
                        
                        E_prev = E_current;
                        E_current = (1/2) * sum((Y_actual - y2)' * (Y_actual - y2));
                        % update the parameters in then-part of second layer
                        q_g(2, :, :) = q_g(2, :, :) - etha * ((-1 * H_dp) * (Y_actual' - y2))';                       
                        t = t + 1;
                    end             
                    y_dp = y2;
                case 3
                    for i = 1 : N
                        for k = 1 : K(dp)
                            mu_y1(k, i) = prod(V_dp(k, d+1, i));
                            mu_y2(k, i) = prod(V_dp(k, d+1:d+2, i));
                        end
                    end
                    t = 1;
                    while (E_current - E_prev > epsilon) & (t < IterMax)
                        A1 = zeros(N, 1);
                        A2 = zeros(N, 1);
                        A3 = zeros(N, 1);
                        
                        for i = 1 : N
%                             A1(i, 1) = sum(q_g(1, :, 1) * H_dp(:, i)); % in q_g the first coefficient is rL_k
%                             A2(i, 1) = sum(q_g(2, :, 1) * mu_y1(:, i) * H_dp(:, i)); % first coefficients in second layer are r2_k
                            A3(i, 1) = sum(q_g(3, :, 1) * mu_y2(:, i) * H_dp(:, i));
%                             B2_1(i, 1) = sum(q_g(2, :, 2) * mu_y1(:, i) * H_dp(:, i));
                            size(q_g(3, :, 2))
                            size(mu_y1(:, i))
                            size(H_dp(:, i))
                            B3_1(i, 1) = sum(q_g(3, :, 2) * mu_y1(:, i) * H_dp(:, i));
                            B3_2(i, 1) = sum(q_g(3, :, 3) * mu_y2(:, i) * H_dp(:, i));
                        end
                        y3 = A3 + A1 .* B3_1 + A2 .* B3_2 + A1 .* B2_1 .* B3_2;
                        
                        E_prev = E_current;
                        E_current = (1/2) * sum((Y_actual - y3)' * (Y_actual - y3));
                        q_g(3, :, :) = q_g(3, :, :) - etha * ((-1 * H_dp) * (Y_actual' - y3))';                       
                        t = t + 1;
                    end
                    y_dp = y3;
            end           
    end
    save G_dp sigma_G_dp c_dp theta_dp omega_dp q_g 
end

