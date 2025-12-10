using Test
import automated_trading_test_environment  as ATTE# Import your package

#add sets of tests under the @testset macro and individual tests under the @test macro
#When adding new unit tests and new external packages to your package updates to the ci.yml file may be needed
@testset "automated_trading_test_environment.jl Tests" begin

    #load test data set
    OHLCVT=Dict()
    pair="ETHUSD"
    interval="1440"
    push!(OHLCVT,pair=>Dict(interval=>Dict("df"=>ATTE.get_pair_interval_df(pair,interval),"dict"=>Dict())))

    #calculate clustering indicators
    OHLCVT["ETHUSD"]["1440"]["df"]=ATTE.calculate_clustering_indicators(OHLCVT["ETHUSD"]["1440"]["df"])

    OHLCVT_ETH_DF=OHLCVT["ETHUSD"]["1440"]["df"]


    #=
    1) random calc of raw values 
    ln return (1 day and 21 day)
    =#
    
    println("Testing 1 day log return...")
    for (t,T) in [(69,1),(149,1),(1512,1),(2289,1),(3519,1)]
        result=isapprox(
            # Actual pre-calculated log return
            OHLCVT_ETH_DF[!, :lnReturn1day][t], 
            
            # Expected log return calculated on the fly
            log(OHLCVT_ETH_DF[!, :close][t] / OHLCVT_ETH_DF[!, :close][t-T]), 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T day log returns step $t passed.")
            @test true==result
        else
            println("$T day log returns step $t failed.")
            @test true==result
        end
    end

    println("Testing 21 day log return...")
    for (t,T) in [(69,21),(149,21),(1512,21),(2289,21),(3519,21)]
        result=isapprox(
            # Actual pre-calculated log return
            OHLCVT_ETH_DF[!, :lnReturn21day][t], 
            
            # Expected log return calculated on the fly
            log(OHLCVT_ETH_DF[!, :close][t] / OHLCVT_ETH_DF[!, :close][t-T]), 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T day log returns step $t passed.")
            @test true==result
        else
            println("$T day log returns step $t failed.")
            @test true==result
        end
    end

    println("Testing 21 day ROC...")
    for (t,T) in [(69,21),(149,21),(1512,21),(2289,21),(3519,21)]
        result=isapprox(
            # Actual pre-calculated ROC
            OHLCVT_ETH_DF[!, :roc21day][t], 
            
            # ROC calculated on the fly
            (OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t]-OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-T])/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-T], 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T day ROC step $t passed.")
            @test true==result
        else
            println("$T day ROC step $t failed.")
            @test true==result
        end
    end

    println("Testing 63 day ROC...")
    for (t,T) in [(69,63),(149,63),(1512,63),(2289,63),(3519,63)]
        result=isapprox(
            # Actual pre-calculated ROC
            OHLCVT_ETH_DF[!, :roc63day][t], 
            
            # ROC calculated on the fly
            (OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t]-OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-T])/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-T], 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T day ROC step $t passed.")
            @test true==result
        else
            println("$T day ROC step $t failed.")
            @test true==result
        end
    end

    
    println("Testing 5 day Realized Volatility...")
    for (t,T) in [(69,5),(149,5),(1512,5),(2289,5),(3519,5)]
        result=isapprox(
            # Actual pre-calculated Realized Volatility
            OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol5day][t], 
            
            # Realized Volatility calculated on the fly
            sqrt(sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn1day][t-T+1:t].^2)/T*252), 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T day Realized Volatility step $t passed.")
            @test true==result
        else
            println("$T day Realized Volatility step $t failed.")
            @test true==result
        end
    end

    println("Testing 10 day Realized Volatility...")
    for (t,T) in [(69,10),(149,10),(1512,10),(2289,10),(3519,10)]
        result=isapprox(
            # Actual pre-calculated Realized Volatility
            OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol10day][t], 
            
            # Realized Volatility calculated on the fly
            sqrt(sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn1day][t-T+1:t].^2)/T*252), 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T day Realized Volatility step $t passed.")
            @test true==result
        else
            println("$T day Realized Volatility step $t failed.")
            @test true==result
        end
    end

    println("Testing 21 day Realized Volatility...")
    for (t,T) in [(69,21),(149,21),(1512,21),(2289,21),(3519,21)]
        result=isapprox(
            # Actual pre-calculated Realized Volatility
            OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol21day][t], 
            
            # Realized Volatility calculated on the fly
            sqrt(sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn1day][t-T+1:t].^2)/T*252), 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T day Realized Volatility step $t passed.")
            @test true==result
        else
            println("$T day Realized Volatility step $t failed.")
            @test true==result
        end
    end

    println("Testing 21 day Garman–Klass volatility...")
    for (T,n) in [(69,21),(149,21),(1512,21),(2289,21),(3519,21)]

        gkt=[]
        for t in T-n:1:T-1
            # Correct Assignment
            high_price  = OHLCVT["ETHUSD"]["1440"]["df"][!,:high][t] # Assign to a new variable name
            low_price   = OHLCVT["ETHUSD"]["1440"]["df"][!,:low][t]
            close_price = OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t]
            open_price  = OHLCVT["ETHUSD"]["1440"]["df"][!,:open][t]

            # Use the correct variable names in the formula
            log_hl = log(high_price / low_price)
            log_co = log(close_price / open_price)
            const_factor = 2*log(2) - 1

            # Calculation: Variance contribution for one day
            variance_contribution = 0.5 * log_hl^2 - const_factor * log_co^2

            push!(gkt, variance_contribution)
        end
        final_gk_variance = sqrt((sum(gkt) / (n))*252)

        result=isapprox(
            # Actual pre-calculated Garman–Klass volatility
            OHLCVT["ETHUSD"]["1440"]["df"][!,:gkVol21day][T], 
            
            # Garman–Klass volatility calculated on the fly
            final_gk_variance, 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$n day Garman–Klass volatility step $T passed.")
            @test true==result
        else
            println("$n day Garman–Klass volatility step $T failed.")
            @test true==result
        end
    end

    println("Testing 21 day Volatility of Volatility...")
    for (t,T) in [(69,21),(149,21),(1512,21),(2289,21),(3519,21)]
        vol=ATTE.daily_realized_volatility(OHLCVT["ETHUSD"]["1440"]["df"][!,:close],1)
        vov=ATTE.daily_realized_volatility(vol,T)
        result=isapprox(
            # Actual pre-calculated Volatility of Volatility
            OHLCVT["ETHUSD"]["1440"]["df"][!,:volOfVol21day][t], 
            
            # Volatility of Volatility calculated on the fly
            vov[t], 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T day Volatility of Volatility step $t passed.")
            @test true==result
        else
            println("$T day Volatility of Volatility step $t failed.")
            @test true==result
        end
    end

    println("Testing 10-50 day Moving Average difference...")
    for (t,T0,T1) in [(69,10,50),(149,10,50),(1512,10,50),(2289,10,50),(3519,10,50)]

        result=isapprox(
            # Actual pre-calculated 10-50 day Moving Average
            OHLCVT["ETHUSD"]["1440"]["df"][!,:maDiff10_50day][t], 
            
            # 10-50 day Moving Average calculated on the fly
            sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-T0+1:t])/T0-sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-T1+1:t])/T1, 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T0 - $T1 day Moving Average difference step $t passed.")
            @test true==result
        else
            println("$T0 - $T1 day Moving Average difference step $t failed.")
            @test true==result
        end
    end

    println("Testing 20 - 100 day Moving Average difference...")
    for (t,T0,T1) in [(149,20,100),(433,20,100),(1512,20,100),(2289,20,100),(3519,20,100)]

        result=isapprox(
            # Actual pre-calculated 20-100 day Moving Average
            OHLCVT["ETHUSD"]["1440"]["df"][!,:maDiff20_100day][t], 
            
            # 20 - 100 day Moving Average calculated on the fly
            sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-T0+1:t])/T0-sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-T1+1:t])/T1, 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T0 - $T1 day Moving Average difference step $t passed.")
            @test true==result
        else
            println("$T0 - $T1 day Moving Average difference step $t failed.")
            @test true==result
        end
    end

    println("Testing Z score...")
    for (t,T) in [(69,21),(149,21),(1512,21),(2289,21),(3519,21)]

        vt=OHLCVT["ETHUSD"]["1440"]["df"][!,:volume][t]
        vm=ATTE.mean(OHLCVT["ETHUSD"]["1440"]["df"][!,:volume][t-T:t-1])
        vs=ATTE.std(OHLCVT["ETHUSD"]["1440"]["df"][!,:volume][t-T:t-1])
        zscore=(vt-vm)/vs

        result=isapprox(
            # Actual pre-calculated zscore
            OHLCVT["ETHUSD"]["1440"]["df"][!,:volumeZscore21day][t], 
            
            # Z score calculated on the fly
            zscore, 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T day Z score step $t passed.")
            @test true==result
        else
            println("$T day Z score step $t failed.")
            @test true==result
        end
    end

    println("Testing Amihud illiquidity...")
    for (t,T) in [(69,21),(149,21),(1512,21),(2289,21),(3519,21)]

        simple_returns=abs.((OHLCVT["ETHUSD"]["1440"]["df"][!,:close][2:end] .- OHLCVT["ETHUSD"]["1440"]["df"][!,:close][1:end-1]))./OHLCVT["ETHUSD"]["1440"]["df"][!,:close][1:end-1]
        illiq = simple_returns ./ (OHLCVT["ETHUSD"]["1440"]["df"][!,:volume][2:end] .* OHLCVT["ETHUSD"]["1440"]["df"][!,:close][2:end])

        result=isapprox(
            # Actual pre-calculated Amid illiquidity
            OHLCVT["ETHUSD"]["1440"]["df"][!,:amihud21day][t], 
            
            # Amid illiquidity calculated on the fly
            ATTE.mean(illiq[t-T:t-1]), 
            
            # Use an absolute tolerance (atol) to handle small floating-point discrepancies
            atol=1e-8 
        )
        if result
            println("$T day Amihud illiquidity step $t passed.")
            @test true==result
        else
            println("$T day Amihud illiquidity step $t failed.")
            @test true==result
        end
    end

end
