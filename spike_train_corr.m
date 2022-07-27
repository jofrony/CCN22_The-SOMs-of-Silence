% all units in ms

function spike_train_corr(spktime_e, spkindex_e, binSize)
        
   edges                                                 = 0 : binSize: max(spktime_e);

  n_neuro                                          = max(spkindex_e);
    MAT                                                = zeros(n_neuro, length(edges) - 1 );
    
    for i = 1 : n_neuro
        spike_ii                                        = spkindex_e == i;
         n_count                                     = histcounts(spktime_e(spike_ii),edges);
         MAT( i , : )                                  = n_count;
    end

    cc                                                      = corr(MAT');
    cc(cc==1)                                          = NaN;
    
    histb             = 0.005;
    edges            = histb/2:histb:1;
    edges            = [-edges(end:-1:1)  edges];
    X                     = histc(cc(:), edges);
    binC                = edges + histb/2;
    figure; bar(binC,X/sum(X));
    xlabel('Correlation coefficient');
    ylabel('Fraction of E/E correlations');
    miu         = nanmean( cc(:) );
    sigma           = nanstd(cc(:));
    title( sprintf( 'miu=%.3f, std=%.3f', miu, sigma))
    alines(miu,'x','LineStyle','--','Color','k');
    xlim([-1 1])



return;