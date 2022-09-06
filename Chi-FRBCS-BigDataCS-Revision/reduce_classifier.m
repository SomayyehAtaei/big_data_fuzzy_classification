function reduce_classifier(intermKVStore, intermValIter, outKVStore)
    while hasnext(intermValIter)
        add(outKVStore, '1', getnext(intermValIter));
    end
end

