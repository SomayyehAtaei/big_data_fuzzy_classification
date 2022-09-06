function reduce(intermKVStore, intermValIter, outKVStore)
    tic
    %% Delete replicate rules
    i = 1;
    while hasnext(intermValIter)
        RB(i, :) = getnext(intermValIter);
        i = i + 1;
    end
    R = size(RB, 1);
    d = size(RB, 2) - 2;
    deleted = zeros(R, 1);
    for i = 1 : R
        duplicate = 0;
        for u = 1 : i - 1
            if isequal(RB(u, 1:d), RB(i, 1:d)) & deleted(u) == 0 
                duplicate = u;
                break;
            end
        end 
        if duplicate > 0
            if deleted(duplicate) == 0 
                if RB(duplicate, d + 1) == RB(i, d + 1)
                    deleted(duplicate) = 1;
                else
                    if RB(duplicate, d + 2) > RB(i, d + 2)
                        deleted(i) = 1;
                    else
                        deleted(duplicate) = 1;
                    end
                end
            end
        end
    end
    
    for i = 1 : R
        if deleted(i) == 0 
            add(outKVStore, 1, RB(i, :));
        end
    end    
    toc
end

