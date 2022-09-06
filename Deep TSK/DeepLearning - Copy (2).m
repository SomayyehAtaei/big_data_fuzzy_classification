function [] = DeepLearning(X, Y_actual, DP, K)
    d = size(X, 1);
    N = size(X, 2);
    numberOfPartitions = 5;
    epsilon = 1e-5;
    IterMax = 500;
    etha = 0.008;
    theta = zeros(DP, d + DP, numberOfPartitions, max(K)); 
    omega = zeros(DP, d + DP, max(K));
    G = zeros(DP, 1);
    sigma_G = zeros(DP, numberOfPartitions);
    centers = zeros(DP, numberOfPartitions);
    q_g = -1 + 2 * rand(DP, max(K), DP); % layer-rule-coefficient
    
    for dp = 1 : DP
        clear temp mu_y1 mu_y2
        switch dp
            case 2
                X = [X; y1'];
            case 3
                X = [X; y1'; y2'];
        end
        
        %% Randomize number of gaussian membership functions in current depth        
%         temp = round(1 + 2 * rand);
%         switch temp
%             case 1
%                 G(dp) = 3;
%             case 2
%                 G(dp) = 5;
%             case 3
%                 G(dp) = 7;
%         end
        G(dp) = 5;
        
        %% Randomize standard deviation of gaussian membership functions in current depth
        sigma_G(dp, 1:G(dp)) = 0.7 + rand(G(dp), 1);

        %% Determine centers of gaussian membership functions in current depth in an uniform partitioning
        centers(dp, 1:G(dp)) = 0 : (1 / (G(dp) - 1)) : 1;

        %% Randomize rule combination matrix in current depth        
        % if-part of each rule in layer dp is consist of d diemention of input and (dp-1) output of prev layers.                
        for k = 1 : K(dp)
            for j = 1 : (d + dp - 1)
%                 for i = 1 : G(dp)
%                     temp2 = rand;
%                     if temp2 > 0.5
%                         theta(dp, j, i, k) = 1;
%                         break;
%                     end
%                 end
                % for each feature in each rule it's partiotions must determined
%                 if sum(theta(dp, j, 1:G(dp), k)) == 0
                    temp2 = round(1 + (G(dp) - 1) * rand);
                    theta(dp, j, temp2, k) = 1;
%                 end
            end
%             if dp > 1
%                 for z = 1 : dp - 1
%                     temp2 = round(1 + (G(dp) - 1) * rand);
%                     theta(dp, d + z, temp2, K(dp)) = 1;
%                 end
%             end    
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
%                 H(k, i) = min(V(k, (1:d), i));
            end
        end

        %% Learn the parameters in the then-parts in fuzzy rules of the current layer dp using GD
            %% Initialize
            t = 1;
            E_current = 1;
            E_prev = 0;
            
            %% Gradient Descent 
            switch dp
                case 1
                    t = 1;
                    while ((E_current - E_prev) > epsilon) & (t < IterMax)
%                     while (t < IterMax)
                        A1 = zeros(N, 1);                         
                        for i = 1 : N
                            A1(i, 1) = sum(q_g(1, 1:K(dp), 1) * H(:, i));
                        end
                        y1 = A1;                        
                        E_prev = E_current;
                        E_current = (1/2) * ((Y_actual - y1') * (Y_actual - y1')');
                        % update the parameters in then-part of first layer
                        q_g(1, 1:K(dp), 1:dp) = q_g(1, 1:K(dp), 1:dp) - etha * ((-1 * H) * (Y_actual - y1')')';
                        t = t + 1;
                    end
                    y_dp = y1;
                case 2
                    for i = 1 : N
                        for k = 1 : K(dp)
                            mu_y1(k, i) = prod(V(k, d+1, i));
                        end
                    end                    
                    t = 1;
                    while (abs(E_current - E_prev) > epsilon) & (t < IterMax)
                        A1 = zeros(N, 1);
                        A2 = zeros(N, 1);
                        
                        for i = 1 : N
                            A1(i, 1) = sum(q_g(1, 1:K(dp), 1) * H(:, i)); % in q_g the first coefficient is rL_k
                            A2(i, 1) = sum(q_g(2, 1:K(dp), 1) * mu_y1(:, i) * H(:, i)); % first coefficients in second layer are r2_k
                            B2_1(i, 1) = sum(q_g(2, 1:K(dp), 2) * mu_y1(:, i) * H(:, i));
                        end
                        y2 = A2 + A1 .* B2_1;
                        
                        E_prev = E_current;
                        E_current = (1/2) * sum((Y_actual - y2') * (Y_actual - y2')');
                        % update the parameters in then-part of second layer
                        q_g(2, 1:K(dp), 1:dp) = q_g(2, 1:K(dp), 1:dp) - etha * ((-1 * H) * (Y_actual - y2')')';                       
                        t = t + 1;
                    end             
                    y_dp = y2;
                case 3
                    for i = 1 : N
                        for k = 1 : K(dp)
                            mu_y1(k, i) = prod(V(k, d+1, i));
                            mu_y2(k, i) = prod(V(k, d+1:d+2, i));
                        end
                    end
                    t = 1;
                    while (abs(E_current - E_prev) > epsilon) & (t < IterMax)
                        A1 = zeros(N, 1);
                        A2 = zeros(N, 1);
                        A3 = zeros(N, 1);
                        
                        for i = 1 : N
%                             A1(i, 1) = sum(q_g(1, 1:K(dp), 1) * H(:, i)); % in q_g the first coefficient is rL_k
%                             A2(i, 1) = sum(q_g(2, 1:K(dp), 1) * mu_y1(:, i) * H(:, i)); % first coefficients in second layer are r2_k
                            A3(i, 1) = sum(q_g(3, 1:K(dp), 1) * mu_y2(:, i) * H(:, i));
%                             B2_1(i, 1) = sum(q_g(2, 1:K(dp), 2) * mu_y1(:, i) * H(:, i));
                            B3_1(i, 1) = sum(q_g(3, 1:K(dp), 2) * mu_y1(:, i) * H(:, i));
                            B3_2(i, 1) = sum(q_g(3, 1:K(dp), 3) * mu_y2(:, i) * H(:, i));
                        end
                        y3 = A3 + A1 .* B3_1 + A2 .* B3_2 + A1 .* B2_1 .* B3_2;
                        
                        E_prev = E_current;
                        E_current = (1/2) * sum((Y_actual - y3') * (Y_actual - y3')');
                        q_g(3, 1:K(dp), 1:dp) = q_g(3, 1:K(dp), 1:dp) - etha * ((-1 * H) * (Y_actual - y3')')';                       
                        t = t + 1;
                    end
                    y_dp = y3;
            end           
    end
    save HID-TSK-FC.mat G sigma_G centers theta omega q_g
end

