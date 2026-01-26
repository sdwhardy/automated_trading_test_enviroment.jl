function normalize_set(OHLCVT_df;indicator_args)
    
    # Signed log transform for heavy-tailed ROC features
    if haskey(indicator_args,"roc")
        for day in indicator_args["roc"]
            OHLCVT_df[!, Symbol("roc$day"*"day")] .= signed_log.(OHLCVT_df[!, Symbol("roc$day"*"day")])
        end
    end
    
    # Log transform for strictly positive Amihud measure
    if haskey(indicator_args,"amihud")
        for day in indicator_args["amihud"]
            OHLCVT_df[!, Symbol("amihud$day"*"day")] = log.(1.0 .+ OHLCVT_df[!, Symbol("amihud$day"*"day")])
        end
    end

    # 9) log(1+trades)
    OHLCVT_df[!, :lntrades] = log.(1.0 .+ OHLCVT_df[!, :trades])

    return OHLCVT_df

end



function zscore_set(OHLCVT_df)
    
    exclude = [:timestamp, :open, :high, :low, :close,
        :volume, :volumeZscore21day
    ]

    zscore_cols = Symbol.(names(OHLCVT_df, Not(exclude)))
    
    OHLCVT_df, OHLCVT_dict = zscore_df(OHLCVT_df, zscore_cols)
    
    OHLCVT_dict            = mean_and_std(OHLCVT_df, zscore_cols)
    
    return OHLCVT_df, OHLCVT_dict

end