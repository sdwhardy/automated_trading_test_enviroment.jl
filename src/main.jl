    import automated_trading_test_environment as ATTE

    # --------------------------------------------------------------------------------------------------
    # Data definitions
    # --------------------------------------------------------------------------------------------------
    # OHLCVT fields:
    #   Open   : first traded price in the interval
    #   High   : highest traded price
    #   Low    : lowest traded price
    #   Close  : final traded price
    #   Volume : total traded volume
    #   Trades : number of individual trades
    #
    # Supported intervals (minutes):
    #   1, 5, 15, 30, 60, 240, 720 (12h), 1440 (24h)
    #
    # Data coverage:
    #   BTC : 2013-10-06T21:34:00+02:00 → 2025-03-31T23:59:00+02:00
    #   ETH : 2015-08-07T14:03:00+02:00 → 2025-03-31T23:59:00+02:00
    # --------------------------------------------------------------------------------------------------


    ####################################################################################################
    # BTC — Daily interval
    ####################################################################################################

    pair     = "XBTUSD"
    interval = "1440"

    # Load OHLCVT dataframe
    OHLCVT_df = ATTE.get_pair_interval_df(pair, interval)


    # --------------------------------------------------------------------------------------------------
    # Feature engineering for clustering and regime analysis
    # --------------------------------------------------------------------------------------------------

    # 1a) Log returns
    #    ln(P_t / P_{t-n}) for multiple rolling horizons
    OHLCVT_df[!, :lnReturn1day]  = ATTE.logreturn(OHLCVT_df[!, :close], 1)
    OHLCVT_df[!, :lnReturn5day]  = ATTE.logreturn(OHLCVT_df[!, :close], 5)
    OHLCVT_df[!, :lnReturn21day] = ATTE.logreturn(OHLCVT_df[!, :close], 21)


    # 1b) signed Log returns
    #    rt=ln(P_t / P_{t-n}) for multiple rolling horizons
    #    sign(rt)*ln(1+abs(rt))
    OHLCVT_df[!, :slnReturn1day]  = ATTE.signed_logreturn(OHLCVT_df[!, :close], 1)
    OHLCVT_df[!, :slnReturn5day]  = ATTE.signed_logreturn(OHLCVT_df[!, :close], 5)
    OHLCVT_df[!, :slnReturn21day] = ATTE.signed_logreturn(OHLCVT_df[!, :close], 21)

    # 2) Rate of Change (ROC)
    #    (P_t - P_{t-n}) / P_{t-n}
    OHLCVT_df[!, :roc21day] = ATTE.roc(OHLCVT_df[!, :close], 21)
    OHLCVT_df[!, :roc63day] = ATTE.roc(OHLCVT_df[!, :close], 63)


    # 3) Realized volatility (annualized)
    #    σ_n = sqrt(1/(n-1) * Σ (r_i - r̄)^2) * sqrt(252)
    OHLCVT_df[!, :realVol5day]  = ATTE.annualized_realized_volatility(OHLCVT_df[!, :close], 5)
    OHLCVT_df[!, :realVol10day] = ATTE.annualized_realized_volatility(OHLCVT_df[!, :close], 10)
    OHLCVT_df[!, :realVol21day] = ATTE.annualized_realized_volatility(OHLCVT_df[!, :close], 21)


    # 4) Garman–Klass volatility (21-day)
    #    σ² = (1/n) Σ [0.5 ln(H/L)² − (2 ln 2 − 1) ln(C/O)²]
    OHLCVT_df[!, :gkVol21day] = ATTE.gk_volatility(
        OHLCVT_df[!, :open],
        OHLCVT_df[!, :high],
        OHLCVT_df[!, :low],
        OHLCVT_df[!, :close],
        21
    )


    # 5) Volatility-of-volatility
    #    Standard deviation of realized volatility over a 21-day window
    OHLCVT_df[!, :volOfVol21day] = ATTE.vol_of_vol(OHLCVT_df[!, :close], 1, 21)


    # 6a) Moving-average differentials
    OHLCVT_df[!, :maDiff10_50day]  = ATTE.ma_diff(OHLCVT_df[!, :close], 10, 50)
    OHLCVT_df[!, :maDiff20_100day] = ATTE.ma_diff(OHLCVT_df[!, :close], 20, 100)


    # 6b) Short term slope
    OHLCVT_df[!, :stSlope3day]=ATTE.short_term_slope(OHLCVT_df[!, :close], 3)
    OHLCVT_df[!, :stSlope10day]=ATTE.short_term_slope(OHLCVT_df[!, :close], 10)
    OHLCVT_df[!, :stSlope21day]=ATTE.short_term_slope(OHLCVT_df[!, :close], 21)


    # 7) Volume Z-score
    #    Z_n = (V_t − mean_n) / std_n
    OHLCVT_df[!, :volumeZscore21day] = ATTE.volume_zscore(OHLCVT_df[!, :volume], 21)


    # 8) Amihud illiquidity (21-day average)
    #    |P_t − P_{t−1}| / (P_{t−1} * P_t * V_t)
    OHLCVT_df[!, :amihud21day] = ATTE.amihud(
        OHLCVT_df[!, :close],
        OHLCVT_df[!, :volume],
        21
    )


    # 9) EMA difference
    OHLCVT_df[!, :ema5MinusEma21] = ATTE.ema_diff(OHLCVT_df[!, :close], 5, 21)#exponential moving average
    OHLCVT_df[!, :ema21MinusEma100] = ATTE.ema_diff(OHLCVT_df[!, :close], 21, 100)

    # 10) EMA slope
    OHLCVT_df[!, :ema5daySlope] = ATTE.ema_slope_normalized(OHLCVT_df[!, :close], 5)
    OHLCVT_df[!, :ema10daySlope] = ATTE.ema_slope_normalized(OHLCVT_df[!, :close], 10)
    OHLCVT_df[!, :ema21daySlope] = ATTE.ema_slope_normalized(OHLCVT_df[!, :close], 21)


    # --------------------------------------------------------------------------------------------------
    # Pre-normalization nonlinear transforms
    # --------------------------------------------------------------------------------------------------

    # Signed log transform for heavy-tailed ROC features
    OHLCVT_df[!, :roc21day] .= ATTE.signed_log.(OHLCVT_df[!, :roc21day])
    OHLCVT_df[!, :roc63day] .= ATTE.signed_log.(OHLCVT_df[!, :roc63day])

    # Log transform for strictly positive Amihud measure
    OHLCVT_df[!, :amihud21day] = log.(1.0 .+ OHLCVT_df[!, :amihud21day])

    # 9) log(1+trades)
    OHLCVT_df[!, :lntrades] = log.(1.0 .+ OHLCVT_df[!, :trades])


    # --------------------------------------------------------------------------------------------------
    # Z-score normalization
    # --------------------------------------------------------------------------------------------------

    exclude = [
        :timestamp, :open, :high, :low, :close,
        :volume, :volumeZscore21day
    ]

    cols = Symbol.(names(OHLCVT_df, ATTE.Not(exclude)))

    OHLCVT_df, OHLCVT_dict = ATTE.zscore_df(OHLCVT_df, cols)
    OHLCVT_dict            = ATTE.mean_and_std(OHLCVT_df, cols)


    # --------------------------------------------------------------------------------------------------
    # Feature validation and PCA
    # --------------------------------------------------------------------------------------------------

    feature_cols = [
        :lnReturn1day,
        :lnReturn5day,
        :lnReturn21day,
        :slnReturn1day,
        :slnReturn5day,
        :slnReturn21day,
        :roc21day,
        :roc63day,
        :realVol5day,
        :realVol10day,
        :realVol21day,
        :gkVol21day,
        :volOfVol21day,
        :maDiff10_50day,
        :maDiff20_100day,
        :ema5MinusEma21,
        :ema21MinusEma100,
        :ema5daySlope,
        :ema10daySlope,
        :ema21daySlope,
        :volumeZscore21day,
        :amihud21day,
        :lntrades
    ]

    results  = ATTE.validate_features(OHLCVT_df, feature_cols)
    pca_dict = ATTE.PCA(OHLCVT_df, feature_cols)

    keep = pca_dict["eigen_values"] .> 1e-6
    pca_dict["pca_df"] = pca_dict["pca_df"][:, keep]

    df_pca = ATTE.percent_explained_PCA(
        pca_dict["pca_df"],
        pca_dict["eigen_values"],
        pca_dict["nonmissing_idx"];
        percent_explained=0.95
    )

    X = Matrix(df_pca[:, ATTE.Not(:row_idx)])

    # Gaussian mixture model clustering 
    best_k, best_gmm, best_bics = ATTE.select_k_bic(X; ks=2:3, kind=:diag)

    post, _ = ATTE.gmmposterior(best_gmm, X)

    best_labels = argmax.(eachrow(post))

    entropy = ATTE.mean_entropy(post)# must be <0.3

    m,s = ATTE.bootstrap_stability(X, best_k, best_labels; B=50)

    println("Entropy: ", entropy)

    for _k = 1:1:best_k
        println(_k,": ", count(x -> x == _k, best_labels))#669 
    end

    println("Stability Mean: ", m, ", Std: ",s)

    clusters = ATTE.data_into_clusters(OHLCVT_df, df_pca, best_labels)

    table_of_means = ATTE.indicator_means_df(clusters,feature_cols)

    

    direction_cols = [
        :lnReturn1day,
        :lnReturn5day,
        :lnReturn21day,
        :slnReturn1day,
        :slnReturn5day,
        :slnReturn21day,
        :roc21day,
        :roc63day
    ]

    table_of_means = ATTE.mean_of_means(:direction,direction_cols,table_of_means)

    volatility_cols = [
        :realVol5day,
        :realVol10day,
        :realVol21day,
        :gkVol21day
    ]

    table_of_means = ATTE.mean_of_means(:volatility,volatility_cols,table_of_means)

    vol_of_vol_cols = [
        :volOfVol21day
    ]

    table_of_means = ATTE.mean_of_means(:vol_of_vol,vol_of_vol_cols,table_of_means)


    trend_cols = [
        :maDiff10_50day,
        :maDiff20_100day,
        :ema5MinusEma21,
        :ema21MinusEma100,
        :ema5daySlope,
        :ema10daySlope,
        :ema21daySlope
    ]

    table_of_means = ATTE.mean_of_means(:trend,trend_cols,table_of_means)

    liquidity_cols = [
        :volumeZscore21day,
        :amihud21day,
        :lntrades
    ]

    table_of_means = ATTE.mean_of_means(:liquidity,liquidity_cols,table_of_means)

    story_of_means_df = ATTE.story_of_means(table_of_means)



    new_max_k=length(unique(eachrow(story_of_means_df[!,ATTE.Not(:k)])))






    
    # Gaussian mixture model clustering 
    best_k, best_gmm, best_bics = ATTE.select_k_bic(X; ks=2:new_max_k, kind=:diag)

    post, _ = ATTE.gmmposterior(best_gmm, X)

    best_labels = argmax.(eachrow(post))

    entropy = ATTE.mean_entropy(post)# must be <0.3

    m,s = ATTE.bootstrap_stability(X, best_k, best_labels; B=50)

    println("Entropy: ", entropy)

    for _k = 1:1:best_k
        println(_k,": ", count(x -> x == _k, best_labels))#669 
    end

    println("Stability Mean: ", m, ", Std: ",s)

    clusters = ATTE.data_into_clusters(OHLCVT_df, df_pca, best_labels)

    table_of_means = ATTE.indicator_means_df(clusters,feature_cols)

    

    direction_cols = [
        :lnReturn1day,
        :lnReturn5day,
        :lnReturn21day,
        :slnReturn1day,
        :slnReturn5day,
        :slnReturn21day,
        :roc21day,
        :roc63day
    ]

    table_of_means = ATTE.mean_of_means(:direction,direction_cols,table_of_means)

    volatility_cols = [
        :realVol5day,
        :realVol10day,
        :realVol21day,
        :gkVol21day
    ]

    table_of_means = ATTE.mean_of_means(:volatility,volatility_cols,table_of_means)

    vol_of_vol_cols = [
        :volOfVol21day
    ]

    table_of_means = ATTE.mean_of_means(:vol_of_vol,vol_of_vol_cols,table_of_means)


    trend_cols = [
        :maDiff10_50day,
        :maDiff20_100day,
        :ema5MinusEma21,
        :ema21MinusEma100,
        :ema5daySlope,
        :ema10daySlope,
        :ema21daySlope
    ]

    table_of_means = ATTE.mean_of_means(:trend,trend_cols,table_of_means)

    liquidity_cols = [
        :volumeZscore21day,
        :amihud21day,
        :lntrades
    ]

    table_of_means = ATTE.mean_of_means(:liquidity,liquidity_cols,table_of_means)

    story_of_means_df = ATTE.story_of_means(table_of_means)


    # X: n × d data matrix
    # labels: vector of cluster assignments from GMM

    






    ################################################### plotting

    
    ATTE.plotlyjs()
    for cols in names(df_pca, ATTE.Not(:row_idx))
        p = ATTE.plot() 
        palette = ATTE.palette(:tab10)

        y_max=abs(maximum(df_pca[!,cols]))
        y_min=abs(minimum(df_pca[!,cols]))
        y_mnmx=maximum([y_min,y_max])
        for k in keys(clusters)
            
            for i = 1:1:length(clusters[k]["pca"][!,:row_idx])
                #if (clusters[k]["pca"][!,:row_idx][i]+1==clusters[k]["pca"][!,:row_idx][i+1])

                ATTE.plot!(
                        p,
                        [clusters[k]["pca"][!,:row_idx][i], clusters[k]["pca"][!,:row_idx][i]+1],
                        [0, 0],                   # y base
                        ribbon=[y_mnmx, y_mnmx], # height of the band
                        color=palette[k],
                        linecolor = palette[k],
                        alpha=0.2,
                        label="",                    
                        legend = false)
                #end

            end

        end
        ATTE.plot!(
            p,
            df_pca[!,:row_idx],
            df_pca[!,cols],
            seriestype = :line,
            color = :black,
            linecolor = :black,
            alpha = 1.0,
            label = string(cols)
        )
        ATTE.display(p)
    end







    ############################################################
    # Transpose so columns = observations
    Xt = transpose(X)
    # Center the data
    Xc = Xt .- ATTE.mean(Xt, dims=1)

    # PCA to 2D
    using MultivariateStats
    pca_model =  ATTE.fit(ATTE.PCA, Xc; maxoutdim=2)
    X_proj = MultivariateStats.transform(pca_model, Xc)

    X_proj = transpose(X_proj)  # now n × 2

    @assert size(X_proj, 1) == 2
@assert size(X_proj, 1) == length(best_labels)

x = vec(X_proj[:,1])#[1, :])
y = vec(X_proj[:,2])#, :])
    # Scatter plot colored by cluster
    ATTE.scatter(
        x, y,
        group=best_labels,
        legend=:topright,
        title="Cluster Visualization (PCA 2D)",
        xlabel="PC1", ylabel="PC2",
        markersize=5
    )

@show size(X_proj)
@show length(best_labels)

 typeof(best_labels)#Vector{Int64}
eltype(best_labels)#Int64
#=ts = OHLCVT["XBTUSD"]["1440"]["df"][!,:timestamp][1]
dt = unix2datetime(ts)
dt_be = ZonedDateTime(dt, tz"Europe/Brussels")
println(dt_be)=#

#=for col in cols
    _mn=minimum(collect(skipmissing(OHLCVT["XBTUSD"]["1440"]["df"][!,col])))
    _mx=maximum(collect(skipmissing(OHLCVT["XBTUSD"]["1440"]["df"][!,col])))
    println("Range of ", col, " is: ", _mn, " - ", _mx)
end=#

