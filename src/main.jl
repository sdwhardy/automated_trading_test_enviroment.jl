    #import automated_trading_test_environment as ATTE
    include("automated_trading_test_environment.jl")
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
    OHLCVT_df = get_pair_interval_df(pair, interval)

    
    indicator_args=Dict(    
        "logreturn" => [1,5,21],    
        "signed_logreturn" => [1,5,21],    
        "roc" => [21,63],    
        "annualized_realized_volatility" => [5,10,21],    
        "gk_volatility" => [21],
        "vol_of_vol" => [(1, 21)],
        "ma_diff" => [(10, 50), (20, 100)],   
        "short_term_slope" => [3,10,21],
        "volume_zscore" => [21],
        "amihud" => [21],
        "ema_diff" => [(5,21),(21,100)],
        "ema_slope_normalized" => [5, 10, 21]
    )

    # --------------------------------------------------------------------------------------------------
    # Calculating financial indicators
    # --------------------------------------------------------------------------------------------------
    OHLCVT_df = indicator_set(OHLCVT_df;indicator_args)

    # --------------------------------------------------------------------------------------------------
    # Pre-normalization nonlinear transforms
    # --------------------------------------------------------------------------------------------------
    OHLCVT_df = normalize_set(OHLCVT_df;indicator_args)

    # --------------------------------------------------------------------------------------------------
    # Z-score normalization
    # --------------------------------------------------------------------------------------------------
    OHLCVT_df, OHLCVT_dict = zscore_set(OHLCVT_df)

    # --------------------------------------------------------------------------------------------------
    # Feature validation and PCA
    # --------------------------------------------------------------------------------------------------    
    exclude = [
        :timestamp, :open, :high, :low, :close,
        :volume, :trades
    ]

    pca_cols = Symbol.(names(OHLCVT_df, Not(exclude)))

    df_pca = pca_set(OHLCVT_df, pca_cols; percent_explained=0.95)

    # --------------------------------------------------------------------------------------------------
    # GMM Clustering
    # --------------------------------------------------------------------------------------------------
    gmm_result = gmm_cluster_set(df_pca, OHLCVT_df, pca_cols)
    
    collapsed_cluster = collapse_clusters(gmm_result["clusters"], gmm_result["table_of_means"], pca_cols)

    ####################################################################################################
    # BTC — Hourly interval
    ####################################################################################################

    # Load OHLCVT dataframe
    interval = "60"
    OHLCVT60_df = get_pair_interval_df(pair, interval)

    # --------------------------------------------------------------------------------------------------
    # Calculating financial indicator set
    # --------------------------------------------------------------------------------------------------
    OHLCVT60_df = indicator_set(OHLCVT60_df;indicator_args)

    # --------------------------------------------------------------------------------------------------
    # Pre-normalization nonlinear transforms
    # --------------------------------------------------------------------------------------------------
    OHLCVT60_df = normalize_set(OHLCVT60_df;indicator_args)

    # --------------------------------------------------------------------------------------------------
    # Z-score normalization
    # --------------------------------------------------------------------------------------------------
    OHLCVT60_df, OHLCVT60_dict = zscore_set(OHLCVT60_df)

    OHLCVT60_day_clusters = sort_higherFreq_into_clusters(collapsed_cluster, OHLCVT60_df)

    OHLCVT60_hourly_clusters=Dict()
 
    for cluster in keys(OHLCVT60_day_clusters)

        # --------------------------------------------------------------------------------------------------
        # Feature validation and PCA
        # --------------------------------------------------------------------------------------------------    
        exclude = [
        :timestamp, :open, :high, :low, :close,
        :volume, :trades
        ]

        pca_cols = Symbol.(names(OHLCVT_df, Not(exclude)))
        
        df_pca = pca_set(OHLCVT60_day_clusters[cluster], pca_cols; percent_explained=0.95)

        # --------------------------------------------------------------------------------------------------
        # k-means cluster set
        # --------------------------------------------------------------------------------------------------
        best_k = best_k4k_means(df_pca)

        kmeans_cluster = k_means_set(df_pca, best_k, OHLCVT60_day_clusters[cluster])

        table_of_means, story_of_means_df = calculate_cluster_means(kmeans_cluster,pca_cols)

        collapsed_cluster = collapse_clusters(kmeans_cluster, table_of_means, pca_cols)
        
        push!(OHLCVT60_hourly_clusters,cluster=>Dict("collapsed_cluster"=>collapsed_cluster,"story_of_means_df"=>story_of_means_df))
  
    end

    for cluster in keys(OHLCVT60_hourly_clusters) 
        println(OHLCVT60_hourly_clusters[cluster]["story_of_means_df"])
    end





















    # --------------------------------------------------------------------------------------------------
    # Feature engineering for clustering and regime analysis
    # --------------------------------------------------------------------------------------------------

    # 1a) Log returns
    #    ln(P_t / P_{t-n}) for multiple rolling horizons
    OHLCVT_df[!, :lnReturn1day]  = logreturn(OHLCVT_df[!, :close], 1)
    OHLCVT_df[!, :lnReturn5day]  = logreturn(OHLCVT_df[!, :close], 5)
    OHLCVT_df[!, :lnReturn21day] = logreturn(OHLCVT_df[!, :close], 21)


    # 1b) signed Log returns
    #    rt=ln(P_t / P_{t-n}) for multiple rolling horizons
    #    sign(rt)*ln(1+abs(rt))
    OHLCVT_df[!, :slnReturn1day]  = signed_logreturn(OHLCVT_df[!, :close], 1)
    OHLCVT_df[!, :slnReturn5day]  = signed_logreturn(OHLCVT_df[!, :close], 5)
    OHLCVT_df[!, :slnReturn21day] = signed_logreturn(OHLCVT_df[!, :close], 21)

    # 2) Rate of Change (ROC)
    #    (P_t - P_{t-n}) / P_{t-n}
    OHLCVT_df[!, :roc21day] = roc(OHLCVT_df[!, :close], 21)
    OHLCVT_df[!, :roc63day] = roc(OHLCVT_df[!, :close], 63)


    # 3) Realized volatility (annualized)
    #    σ_n = sqrt(1/(n-1) * Σ (r_i - r̄)^2) * sqrt(252)
    OHLCVT_df[!, :realVol5day]  = annualized_realized_volatility(OHLCVT_df[!, :close], 5)
    OHLCVT_df[!, :realVol10day] = annualized_realized_volatility(OHLCVT_df[!, :close], 10)
    OHLCVT_df[!, :realVol21day] = annualized_realized_volatility(OHLCVT_df[!, :close], 21)


    # 4) Garman–Klass volatility (21-day)
    #    σ² = (1/n) Σ [0.5 ln(H/L)² − (2 ln 2 − 1) ln(C/O)²]
    OHLCVT_df[!, :gkVol21day] = gk_volatility(
        OHLCVT_df[!, :open],
        OHLCVT_df[!, :high],
        OHLCVT_df[!, :low],
        OHLCVT_df[!, :close],
        21
    )


    # 5) Volatility-of-volatility
    #    Standard deviation of realized volatility over a 21-day window
    OHLCVT_df[!, :volOfVol21day] = vol_of_vol(OHLCVT_df[!, :close], 1, 21)


    # 6a) Moving-average differentials
    OHLCVT_df[!, :maDiff10_50day]  = ma_diff(OHLCVT_df[!, :close], 10, 50)
    OHLCVT_df[!, :maDiff20_100day] = ma_diff(OHLCVT_df[!, :close], 20, 100)


    # 6b) Short term slope
    OHLCVT_df[!, :stSlope3day]=short_term_slope(OHLCVT_df[!, :close], 3)
    OHLCVT_df[!, :stSlope10day]=short_term_slope(OHLCVT_df[!, :close], 10)
    OHLCVT_df[!, :stSlope21day]=short_term_slope(OHLCVT_df[!, :close], 21)


    # 7) Volume Z-score
    #    Z_n = (V_t − mean_n) / std_n
    OHLCVT_df[!, :volumeZscore21day] = volume_zscore(OHLCVT_df[!, :volume], 21)


    # 8) Amihud illiquidity (21-day average)
    #    |P_t − P_{t−1}| / (P_{t−1} * P_t * V_t)
    OHLCVT_df[!, :amihud21day] = amihud(
        OHLCVT_df[!, :close],
        OHLCVT_df[!, :volume],
        21
    )


    # 9) EMA difference
    OHLCVT_df[!, :ema5MinusEma21] = ema_diff(OHLCVT_df[!, :close], 5, 21)#exponential moving average
    OHLCVT_df[!, :ema21MinusEma100] = ema_diff(OHLCVT_df[!, :close], 21, 100)

    # 10) EMA slope
    OHLCVT_df[!, :ema5daySlope] = ema_slope_normalized(OHLCVT_df[!, :close], 5)
    OHLCVT_df[!, :ema10daySlope] = ema_slope_normalized(OHLCVT_df[!, :close], 10)
    OHLCVT_df[!, :ema21daySlope] = ema_slope_normalized(OHLCVT_df[!, :close], 21)

     # Signed log transform for heavy-tailed ROC features
    OHLCVT_df[!, :roc21day] .= signed_log.(OHLCVT_df[!, :roc21day])
    OHLCVT_df[!, :roc63day] .= signed_log.(OHLCVT_df[!, :roc63day])

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

    zscore_cols = Symbol.(names(OHLCVT_df, Not(exclude)))

    OHLCVT_df, OHLCVT_dict = zscore_df(OHLCVT_df, zscore_cols)
    OHLCVT_dict            = mean_and_std(OHLCVT_df, zscore_cols)


    # --------------------------------------------------------------------------------------------------
    # Feature validation and PCA
    # --------------------------------------------------------------------------------------------------    
    exclude = [
        :timestamp, :open, :high, :low, :close,
        :volume, :trades
    ]

    pca_cols = Symbol.(names(OHLCVT_df, Not(exclude)))

    pca_dict = PCA(OHLCVT_df, pca_cols)

    df_pca = percent_explained_PCA(
        pca_dict["pca_df"],
        pca_dict["eigen_values"],
        pca_dict["nonmissing_idx"];
        percent_explained=0.95
    )

    df_pca_functions=deepcopy(df_pca)

    df_pca_functions==df_pca