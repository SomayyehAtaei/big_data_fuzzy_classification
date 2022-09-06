function reduce_prediction(intermKVStore, intermValIter, outKVStore)
    
    TP_global = 0;
    TN_global = 0;
    FP_global = 0;
    FN_global = 0;
    N_global = 0;
    K_global = 0;
    
    while hasnext(intermValIter)
        temp = getnext(intermValIter);
        TP_global = TP_global + temp(1, 1);
        TN_global = TN_global + temp(1, 2);
        FP_global = FP_global + temp(1, 3);
        FN_global = FN_global + temp(1, 4);
        N_global = N_global + temp(1, 5);
        K_global = K_global + temp(1, 6);
    end
    
    accuracy = (TP_global + TN_global) / N_global;
    sensitivity = TP_global / (TP_global + FN_global);    
    specificity = TP_global / (TN_global + FP_global);
    rulesNumber = K_global;
    TPrate = TP_global / (TP_global + FN_global);
    FPrate = FP_global / (FP_global + TN_global);
    
    AUC = (1 + TPrate - FPrate) / 2; 
    add(outKVStore, 1, [AUC accuracy TP_global TN_global FP_global FN_global rulesNumber]);
end

