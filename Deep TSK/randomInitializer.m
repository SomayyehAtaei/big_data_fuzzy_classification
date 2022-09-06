function [G, sigma_G, centers, theta, omega] = randomInitializer(G, sigma_G, centers, theta, omega, K, dp, d)
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

end

