using Test
import automated_trading_test_environment as ATTE

# --------------------------------------------------------------------------------------------------
# Test suite for automated_trading_test_environment.jl
# --------------------------------------------------------------------------------------------------
# Notes:
# - Each indicator is validated against an explicit on-the-fly calculation
# - Absolute tolerances are used to account for floating-point error
# - Tests are intentionally verbose to aid CI debugging
# --------------------------------------------------------------------------------------------------

@testset "automated_trading_test_environment.jl Tests" begin

    # --------------------------------------------------------------------------------------------------
    # Load test dataset (ETH, daily)
    # --------------------------------------------------------------------------------------------------
    OHLCVT = Dict()
    pair     = "ETHUSD"
    interval = "1440"

    push!(
        OHLCVT,
        pair => Dict(
            interval => Dict(
                "df"   => ATTE.get_pair_interval_df(pair, interval),
                "dict" => Dict()
            )
        )
    )

    # Compute all clustering indicators using package API
    OHLCVT["ETHUSD"]["1440"]["df"] =
        ATTE.calculate_clustering_indicators(OHLCVT["ETHUSD"]["1440"]["df"])

    OHLCVT_ETH_DF = OHLCVT["ETHUSD"]["1440"]["df"]


    # --------------------------------------------------------------------------------------------------
    # 1a) Log return validation
    # --------------------------------------------------------------------------------------------------

    println("Testing 1 day log return...")
    for (t, T) in [(69,1), (149,1), (1512,1), (2289,1), (3519,1)]
        result = isapprox(
            OHLCVT_ETH_DF[!, :lnReturn1day][t],
            log(OHLCVT_ETH_DF[!, :close][t] / OHLCVT_ETH_DF[!, :close][t-T]),
            atol = 1e-8
        )
        println("$T day log returns step $t ", result ? "passed." : "failed.")
        @test result == true
    end

    println("Testing 21 day log return...")
    for (t, T) in [(69,21), (149,21), (1512,21), (2289,21), (3519,21)]
        result = isapprox(
            OHLCVT_ETH_DF[!, :lnReturn21day][t],
            log(OHLCVT_ETH_DF[!, :close][t] / OHLCVT_ETH_DF[!, :close][t-T]),
            atol = 1e-8
        )
        println("$T day log returns step $t ", result ? "passed." : "failed.")
        @test result == true
    end

    # --------------------------------------------------------------------------------------------------
    # 1b) signed Log return validation
    # --------------------------------------------------------------------------------------------------

    println("Testing 1 day signed log return...")
    for (t, T) in [(69,1), (149,1), (1512,1), (2289,1), (3519,1)]
        sln=log(OHLCVT_ETH_DF[!, :close][t] / OHLCVT_ETH_DF[!, :close][t-T])
        result = isapprox(
            OHLCVT_ETH_DF[!, :slnReturn1day][t],
            sign(sln)*log(1+abs(sln)),
            atol = 1e-8
        )
        println("$T day signed log returns step $t ", result ? "passed." : "failed.")
        @test result == true
    end

    println("Testing 21 day signed log return...")
    for (t, T) in [(69,21), (149,21), (1512,21), (2289,21), (3519,21)]
        sln=log(OHLCVT_ETH_DF[!, :close][t] / OHLCVT_ETH_DF[!, :close][t-T])
        result = isapprox(
            OHLCVT_ETH_DF[!, :slnReturn21day][t],
            sign(sln)*log(1+abs(sln)),
            atol = 1e-8
        )
        println("$T day signed log returns step $t ", result ? "passed." : "failed.")
        @test result == true
    end


    # --------------------------------------------------------------------------------------------------
    # 2) Rate of Change (ROC)
    # --------------------------------------------------------------------------------------------------

    println("Testing 21 day ROC...")
    for (t, T) in [(69,21), (149,21), (1512,21), (2289,21), (3519,21)]
        result = isapprox(
            OHLCVT_ETH_DF[!, :roc21day][t],
            (OHLCVT_ETH_DF[!, :close][t] - OHLCVT_ETH_DF[!, :close][t-T]) /
            OHLCVT_ETH_DF[!, :close][t-T],
            atol = 1e-8
        )
        println("$T day ROC step $t ", result ? "passed." : "failed.")
        @test result == true
    end

    println("Testing 63 day ROC...")
    for (t, T) in [(69,63), (149,63), (1512,63), (2289,63), (3519,63)]
        result = isapprox(
            OHLCVT_ETH_DF[!, :roc63day][t],
            (OHLCVT_ETH_DF[!, :close][t] - OHLCVT_ETH_DF[!, :close][t-T]) /
            OHLCVT_ETH_DF[!, :close][t-T],
            atol = 1e-8
        )
        println("$T day ROC step $t ", result ? "passed." : "failed.")
        @test result == true
    end


    # --------------------------------------------------------------------------------------------------
    # 3) Realized volatility (annualized)
    # --------------------------------------------------------------------------------------------------

    for T in (5, 10, 21)
        println("Testing $T day Realized Volatility...")
        for t in (69, 149, 1512, 2289, 3519)
            result = isapprox(
                OHLCVT_ETH_DF[!, Symbol("realVol$(T)day")][t],
                sqrt(sum(OHLCVT_ETH_DF[!, :lnReturn1day][(t-T):(t-1)].^2) / T * 252),
                atol = 1e-8
            )
            println("$T day Realized Volatility step $t ", result ? "passed." : "failed.")
            @test result == true
        end
    end


    # --------------------------------------------------------------------------------------------------
    # 4) Garman–Klass volatility (21-day)
    # --------------------------------------------------------------------------------------------------

    println("Testing 21 day Garman–Klass volatility...")
    for (T, n) in [(69,21), (149,21), (1512,21), (2289,21), (3519,21)]

        gkt = Float64[]
        for t in (T-n):(T-1)
            log_hl = log(OHLCVT_ETH_DF[!, :high][t] / OHLCVT_ETH_DF[!, :low][t])
            log_co = log(OHLCVT_ETH_DF[!, :close][t] / OHLCVT_ETH_DF[!, :open][t])
            push!(gkt, 0.5 * log_hl^2 - (2*log(2)-1) * log_co^2)
        end

        final_gk = sqrt(sum(gkt) / n * 252)

        result = isapprox(
            OHLCVT_ETH_DF[!, :gkVol21day][T],
            final_gk,
            atol = 1e-8
        )
        println("$n day Garman–Klass volatility step $T ", result ? "passed." : "failed.")
        @test result == true
    end


    # --------------------------------------------------------------------------------------------------
    # 5) Volatility of volatility
    # --------------------------------------------------------------------------------------------------

    println("Testing 21 day Volatility of Volatility...")
    vol = ATTE.daily_realized_volatility(OHLCVT_ETH_DF[!, :close], 1)

    for (t, T) in [(69,21), (149,21), (1512,21), (2289,21), (3519,21)]
        vov = ATTE.std(collect(skipmissing(vol[(t-T+1):t])))
        result = isapprox(OHLCVT_ETH_DF[!, :volOfVol21day][t], vov, atol=1e-8)
        println("$T day Volatility of Volatility step $t ", result ? "passed." : "failed.")
        @test result == true
    end


    # --------------------------------------------------------------------------------------------------
    # 6) Moving-average differentials
    # --------------------------------------------------------------------------------------------------

    println("Testing moving average differences...")
    for (t, T0, T1) in [(69,10,50), (149,10,50), (1512,10,50), (2289,10,50), (3519,10,50)]
        result = isapprox(
            OHLCVT_ETH_DF[!, :maDiff10_50day][t],
            sum(OHLCVT_ETH_DF[!, :close][t-T0+1:t]) / T0 -
            sum(OHLCVT_ETH_DF[!, :close][t-T1+1:t]) / T1,
            atol = 1e-8
        )
        println("$T0-$T1 MA diff step $t ", result ? "passed." : "failed.")
        @test result == true
    end

    println("Testing 20-100 day MA difference...")
    for (t, T0, T1) in [(149,20,100), (433,20,100), (1512,20,100), (2289,20,100), (3519,20,100)]
        result = isapprox(
            OHLCVT_ETH_DF[!, :maDiff20_100day][t],
            sum(OHLCVT_ETH_DF[!, :close][t-T0+1:t]) / T0 -
            sum(OHLCVT_ETH_DF[!, :close][t-T1+1:t]) / T1,
            atol = 1e-8
        )
        println("$T0-$T1 MA diff step $t ", result ? "passed." : "failed.")
        @test result == true
    end

    # --------------------------------------------------------------------------------------------------
    # 6b) Short term slope
    # --------------------------------------------------------------------------------------------------

    println("Testing 3 day short term slope...")
    for (t, n) in [(69,3), (149,3), (1512,3), (2289,3), (3519,3)]
        indices = 0:(n-1)
        mean_i = ATTE.mean(indices)
        denom = sum((indices .- mean_i).^2)
        window = OHLCVT_ETH_DF[!, :close][(t-n+1):t]
        mean_x = ATTE.mean(window)
        slope = sum((indices .- mean_i) .* (window .- mean_x)) / denom
        result = isapprox(
            OHLCVT_ETH_DF[!, :stSlope3day][t],
            slope,
            atol = 1e-8
        )
        println("$n day short term slope step $t ", result ? "passed." : "failed.")
        @test result == true
    end

    println("Testing 21 day short term slope...")
    for (t, n) in [(69,21), (149,21), (1512,21), (2289,21), (3519,21)]
        indices = 0:(n-1)
        mean_i = ATTE.mean(indices)
        denom = sum((indices .- mean_i).^2)
        window = OHLCVT_ETH_DF[!, :close][(t-n+1):t]
        mean_x = ATTE.mean(window)
        slope = sum((indices .- mean_i) .* (window .- mean_x)) / denom
        result = isapprox(
            OHLCVT_ETH_DF[!, :stSlope21day][t],
            slope,
            atol = 1e-8
        )
        println("$n day short term slope step $t ", result ? "passed." : "failed.")
        @test result == true
    end

    # --------------------------------------------------------------------------------------------------
    # 6c) EMA differentials
    # --------------------------------------------------------------------------------------------------

    println("Testing EMA 5–21 day difference...")
    ema5 = ATTE.exma(OHLCVT_ETH_DF[!, :close], 5)
    ema21 = ATTE.exma(OHLCVT_ETH_DF[!, :close], 21)

    for t in (69, 149, 1512, 2289, 3519)
        result = isapprox(
            OHLCVT_ETH_DF[!, :ema5MinusEma21][t],
            ema5[t] - ema21[t],
            atol = 1e-8
        )
        println("EMA 5-21 diff step $t ", result ? "passed." : "failed.")
        @test result == true
    end

    println("Testing EMA 21–100 day difference...")
    ema21  = ATTE.exma(OHLCVT_ETH_DF[!, :close], 21)
    ema100 = ATTE.exma(OHLCVT_ETH_DF[!, :close], 100)

    for t in (149, 433, 1512, 2289, 3519)
        result = isapprox(
            OHLCVT_ETH_DF[!, :ema21MinusEma100][t],
            ema21[t] - ema100[t],
            atol = 1e-8
        )
        println("EMA 21–100 diff step $t ", result ? "passed." : "failed.")
        @test result == true
    end

    # --------------------------------------------------------------------------------------------------
    # 6d) EMA slope
    # --------------------------------------------------------------------------------------------------

    println("Testing 10 day EMA slope...")
    ema10 = ATTE.exma(OHLCVT_ETH_DF[!, :close], 10)

    for (t, n) in [(69,10), (149,10), (1512,10), (2289,10), (3519,10)]

        slope =  (ema10[t] - ema10[t-1]) / ema10[t-1]

        result = isapprox(
            OHLCVT_ETH_DF[!, :ema10daySlope][t],
            slope,
            atol = 1e-8
        )
        println("10 day EMA slope step $t ", result ? "passed." : "failed.")
        @test result == true
    end

    println("Testing 21 day EMA slope...")
    ema21 = ATTE.exma(OHLCVT_ETH_DF[!, :close], 21)

    for (t, n) in [(69,21), (149,21), (1512,21), (2289,21), (3519,21)]
        
        slope =  (ema21[t] - ema21[t-1]) / ema21[t-1]

        result = isapprox(
            OHLCVT_ETH_DF[!, :ema21daySlope][t],
            slope,
            atol = 1e-8
        )
        println("21 day EMA slope step $t ", result ? "passed." : "failed.")
        @test result == true
    end


    # --------------------------------------------------------------------------------------------------
    # 7) Volume Z-score
    # --------------------------------------------------------------------------------------------------

    println("Testing Z score...")
    for (t, T) in [(69,21), (149,21), (1512,21), (2289,21), (3519,21)]
        z = (OHLCVT_ETH_DF[!, :volume][t] -
             ATTE.mean(OHLCVT_ETH_DF[!, :volume][t-T:t-1])) /
            ATTE.std(OHLCVT_ETH_DF[!, :volume][t-T:t-1])

        result = isapprox(OHLCVT_ETH_DF[!, :volumeZscore21day][t], z, atol=1e-8)
        println("$T day Z score step $t ", result ? "passed." : "failed.")
        @test result == true
    end


    # --------------------------------------------------------------------------------------------------
    # 8) Amihud illiquidity
    # --------------------------------------------------------------------------------------------------

    println("Testing Amihud illiquidity...")
    simple_returns =
        abs.(OHLCVT_ETH_DF[!, :close][2:end] .- OHLCVT_ETH_DF[!, :close][1:end-1]) ./
        OHLCVT_ETH_DF[!, :close][1:end-1]

    illiq =
        simple_returns ./ (OHLCVT_ETH_DF[!, :volume][2:end] .* OHLCVT_ETH_DF[!, :close][2:end])

    for (t, T) in [(69,21), (149,21), (1512,21), (2289,21), (3519,21)]
        result = isapprox(
            OHLCVT_ETH_DF[!, :amihud21day][t],
            ATTE.mean(illiq[t-T:t-1]),
            atol = 1e-8
        )
        println("$T day Amihud step $t ", result ? "passed." : "failed.")
        @test result == true
    end


    # --------------------------------------------------------------------------------------------------
    # 9) Normalization checks
    # --------------------------------------------------------------------------------------------------

    println("Testing Normalization...")
    OHLCVT_ETH_DF[!, :roc21day]   .= ATTE.signed_log.(OHLCVT_ETH_DF[!, :roc21day])
    OHLCVT_ETH_DF[!, :roc63day]   .= ATTE.signed_log.(OHLCVT_ETH_DF[!, :roc63day])
    OHLCVT_ETH_DF[!, :amihud21day] = log.(1.0 .+ OHLCVT_ETH_DF[!, :amihud21day])

    exclude = [:timestamp, :open, :high, :low, :close, :volume, :trades, :volumeZscore21day]
    cols    = Symbol.(names(OHLCVT_ETH_DF, ATTE.Not(exclude)))

    OHLCVT_ETH_DF, OHLCVT["ETHUSD"]["1440"]["dict"] = ATTE.zscore_df(OHLCVT_ETH_DF, cols)
    OHLCVT["ETHUSD"]["1440"]["dict"] = ATTE.mean_and_std(OHLCVT_ETH_DF, cols)

    results = ATTE.validate_features(OHLCVT_ETH_DF, cols)

    @test all(results[!, :mean_ok]) == true
    @test all(results[!, :std_ok])  == true
    @test all(results[!, :aligned]) == true


    # --------------------------------------------------------------------------------------------------
    # 10) PCA validation
    # --------------------------------------------------------------------------------------------------

    println("Testing Principal Component Analysis (PCA)...")

    feature_cols = [
        :lnReturn1day, :lnReturn5day, :lnReturn21day,
        :roc21day, :roc63day,
        :realVol5day, :realVol10day, :realVol21day,
        :gkVol21day, :volOfVol21day,
        :maDiff10_50day, :maDiff20_100day,
        :volumeZscore21day, :amihud21day,
        :ema5MinusEma21,:ema21MinusEma100,
        :ema5daySlope, :ema10daySlope, :ema21daySlope
    ]

    pca_dict = ATTE.PCA(OHLCVT_ETH_DF, feature_cols)
    tol = 1e-8

    # 1) Eigenvalues non-negative
    @test minimum(pca_dict["eigen_values"]) ≥ -tol

    # 2) Eigenvectors orthonormal
    @test ATTE.norm(pca_dict["eigen_vectors"]' * pca_dict["eigen_vectors"] - ATTE.I, Inf) < tol

    # 3) Variance of PC scores equals eigenvalues
    pc_vars = ATTE.vec(ATTE.var(pca_dict["pca_df"]; dims=1))
    @test ATTE.norm(pc_vars - pca_dict["eigen_values"], Inf) < tol

    # 4) Reconstruction accuracy
    X_recon = pca_dict["pca_df"] * pca_dict["eigen_vectors"]'
    @test ATTE.norm(X_recon - pca_dict["X_centered"], Inf) < tol

    df_pca = ATTE.percent_explained_PCA(
        pca_dict["pca_df"],
        pca_dict["eigen_values"],
        pca_dict["nonmissing_idx"];
        percent_explained=0.95
    )

    #Cluster using Gaussian mixture model clustering 
    println("Testing GMM Clustering...")

    # Gaussian mixture model clustering 
    X = Matrix(df_pca[:, ATTE.Not(:row_idx)])

    best_k, best_gmm, best_bics = ATTE.select_k_bic(X; ks=2:15, kind=:diag)

    println("Checking lowest BIC selected...")

    @test best_k == argmin(best_bics)+1

    post, _ = ATTE.gmmposterior(best_gmm, X)

    best_labels = argmax.(eachrow(post))

    println("Checking cluster entropy...")

    entropy = ATTE.mean_entropy(post)

    @test entropy < 0.4

    println("Checking cluster stability...")

    m,s = ATTE.bootstrap_stability(X, best_k, best_labels; B=50)

    @test m > 0.7

    @test s < 0.2

    println("Lowest BIC occurs at k: ",best_k)

    println("Entropy: ", entropy)

    println("Stability Mean: ", m, ", Std: ",s)

end