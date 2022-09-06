function reduce_modelCreation(intermKVStore, intermValIter, outKVStore)
    check = true;

    while hasnext(intermValIter)
        str = getnext(intermValIter);
        load str
        if check
            q_g_global = zeros(DP, max(K), DP); % layer-rule-coefficient    
            theta_global = zeros(DP, d + DP, 7, max(K)); 
            omega_global = zeros(DP, d + DP, max(K));    
            W_global = ones(DP, max(K));
            t = 1;
            o = 1;
            q = 1;
            w = 1;
            check = false;           
        end
        theta_global(:, :, :, t:t -1 + size(theta, 4)) = theta(:, :, :, 1:size(theta, 4));
        t = t + size(theta, 4);
        omega_global(:, :, o:o -1 + size(omega, 3)) = omega(:, :, 1:size(omega, 3));
        o = o + size(omega, 3);
        q_g_global(:, q:q -1 + size(q_g, 3), :) = q_g(:, 1:size(q_g, 3), :);
        q == q + size(q_g, 3);
        W_global(:, w: w - 1 + size(W, 2)) = W(:, 1:size(W, 2));
        w = w + size(W, 2);
    end
    save HID-TSK-FC.mat G sigma_G centers theta omega q_g W
    add(outKVStore, 1, "HID-TSK-FC.mat");
end
