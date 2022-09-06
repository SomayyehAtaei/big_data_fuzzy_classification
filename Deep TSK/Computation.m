function [U, V, H] = Computation(X, G, centers, sigma_G, theta, omega, K, dp, N, d)

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
%                                     V(k, j, i) = 1 - prod(1 - squeeze(theta(dp, j, 1:G(dp), k))' * U(:, j, i));
                    V(k, j, i) = 1 - max(1 - squeeze(theta(dp, j, 1:G(dp), k))' * U(:, j, i));
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

end

