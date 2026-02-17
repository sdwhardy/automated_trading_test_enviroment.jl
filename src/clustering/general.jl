"""
    collapse_clusters(clusters, table_of_means, cols)

Iteratively collapse clusters using a two-stage procedure combining
statistical similarity and semantic equivalence.

The function proceeds as follows:

1. **Statistical collapse**  
   Clusters are first merged based on numerical similarity of their
   feature means using `statistic_collapse_clusters`, with a fixed
   tolerance `eps = 0.3` applied to the selected feature columns `cols`.

2. **Recompute cluster means**  
   After merging, cluster-level means and their semantic summaries
   are recomputed using `calculate_cluster_means`.

3. **Semantic collapse**  
   Clusters with identical semantic signatures (derived from discretized
   high-level indicators such as direction, volatility, trend, etc.)
   are merged using `semantic_collapse_clusters`.

4. **Final recomputation**  
   Cluster means and semantic stories are recomputed once more after
   semantic merging. The final semantic table is printed for inspection.

This process reduces clusters while preserving both statistical coherence
and interpretability.

### Arguments
- `clusters`: Vector or array of cluster assignments for each observation.
- `table_of_means`: Initial DataFrame of cluster-level feature means.
- `cols`: Vector of feature column symbols used for statistical comparison.

### Returns
- `collapsed_cluster`: Updated cluster assignments after statistical and
  semantic collapsing.
"""
function collapse_clusters(clusters, table_of_means, cols)

    result = statistic_collapse_clusters(table_of_means; feature_cols = cols, eps = 0.3)

    collapsed_cluster = merge_clusters(result, clusters)
    
    table_of_means, story_of_means_df = calculate_cluster_means(collapsed_cluster,cols)
    
    result = semantic_collapse_clusters(story_of_means_df)

    collapsed_cluster = merge_clusters(result, collapsed_cluster)
    
    table_of_means, story_of_means_df = calculate_cluster_means(collapsed_cluster, cols)
    
    println(story_of_means_df)

    return collapsed_cluster

end


"""
    merge_clusters(result, best_run)

Merge clusters according to a precomputed cluster merge mapping.

This function applies the `merge_map` contained in `result` to an
existing clustering (`best_run`) and constructs a new dictionary of
collapsed clusters. Clusters that map to the same target cluster ID
are merged by concatenating their underlying data and PCA projections.

For each merged cluster:
- The `"data"` fields are concatenated and sorted by `:timestamp`.
- The `"pca"` fields are concatenated and sorted by `:row_idx`.

The function assumes that each entry in `best_run` is a dictionary-like
object containing at least the keys `"data"` and `"pca"`.

### Arguments
- `result`: An object with a `merge_map::Dict{Int,Int}` defining how
  original cluster IDs should be merged.
- `best_run`: Dictionary of clusters keyed by cluster ID, where each
  value contains cluster-specific data and PCA results.

### Returns
- `collapsed_cluster`: Dictionary of merged clusters keyed by the
  representative cluster IDs defined in `merge_map`.
"""
function merge_clusters(result, best_run)

    collapsed_cluster=Dict()

    for (k,v) in result.merge_map

        if !(haskey(collapsed_cluster,v))

            push!(collapsed_cluster,v=>best_run[k])

        else

            collapsed_cluster[v]["data"]=sort(vcat(collapsed_cluster[v]["data"],best_run[k]["data"]), :timestamp)

            collapsed_cluster[v]["pca"]=sort(vcat(collapsed_cluster[v]["pca"],best_run[k]["pca"]), :row_idx)

        end

    end

    return collapsed_cluster

end



"""
Collapse statistically indistinguishable clusters.

Inputs:
- table_of_means :: DataFrame
- feature_cols   :: Vector{Symbol}
- eps            :: Float64   (merge threshold)

Returns:
- effective_K :: Int
- cluster_map :: Dict{Int,Int}  (original -> collapsed)
"""
function statistic_collapse_clusters(
    table_of_means::DataFrame;
    feature_cols::Vector{Symbol},
    eps::Float64 = 0.30
)
    K = nrow(table_of_means)
    d = length(feature_cols)

    # extract mean matrix
    M = Matrix(table_of_means[:, feature_cols])

    # pairwise distance matrix
    D = zeros(K, K)
    for i in 1:K, j in i+1:K
        D[i, j] = norm(M[i, :] .- M[j, :]) / sqrt(d)
        D[j, i] = D[i, j]
    end

    # union-find for merging
    parent = collect(1:K)

    find(x) = parent[x] == x ? x : (parent[x] = find(parent[x]))
    function union(x, y)
        px, py = find(x), find(y)
        px != py && (parent[py] = px)
    end

    # merge close clusters
    for i in 1:K, j in i+1:K
        if D[i, j] < eps
            union(i, j)
        end
    end

    # build collapsed mapping
    roots = unique(find.(1:K))
    collapsed_id = Dict(r => i for (i, r) in enumerate(roots))

    cluster_map = Dict(i => collapsed_id[find(i)] for i in 1:K)

    return (
        effective_K = length(roots),
        merge_map = cluster_map,
        distance_matrix = D
    )
end


"""
Collapse clusters that are semantically equivalent.

Arguments
---------
table_of_means :: DataFrame
    Rows = clusters
    Must contain semantic columns with values in {-1, 0, +1}
k_col :: Symbol
    Cluster identifier column
semantic_cols :: Vector{Symbol}
    Columns defining regime semantics

Returns
-------
(
  effective_k,
  merge_map,
  semantic_groups
)
"""

function semantic_collapse_clusters(
    table_of_means;
    k_col::Symbol = :k,
    semantic_cols::Vector{Symbol} = [
        :direction, :volatility, :vol_of_vol, :trend, :liquidity
    ]
)
    clusters = table_of_means[:, k_col]

    # semantic signature for each cluster
    signatures = Dict{Int, Tuple}()

    for i in 1:nrow(table_of_means)
        k = clusters[i]
        signatures[k] = Tuple(table_of_means[i, semantic_cols])
    end

    # group clusters by identical semantic signature
    groups = Dict{Tuple, Vector{Int}}()
    for (k, sig) in signatures
        push!(get!(groups, sig, Int[]), k)
    end

    # representative = smallest cluster id (stable)
    merge_map = Dict{Int,Int}()
    for group in values(groups)
        rep = minimum(group)
        for k in group
            merge_map[k] = rep
        end
    end

    semantic_groups = collect(values(groups))

    return (
        effective_k = length(semantic_groups),
        merge_map = merge_map,
        groups = semantic_groups
    )
end

"""
    calculate_cluster_means(clusters, feature_cols)
        -> (table_of_means, story_of_means_df)

Compute cluster-level indicator means and derive semantic aggregate features.

This function aggregates raw indicator features at the cluster level, then
constructs higher-level semantic descriptors (direction, volatility,
vol-of-vol, trend, liquidity) by averaging predefined groups of indicators.
The result is a compact, interpretable representation of each cluster.

# Arguments
- `clusters`: Dictionary-like object indexed by cluster identifier, where each
  entry contains a `"data"` field holding a `DataFrame` of observations.
- `feature_cols`: Vector of indicator column names (`Symbol`) to be aggregated
  at the cluster level.

# Returns
- `table_of_means`: `DataFrame` containing, for each cluster:
  - raw indicator means (from `feature_cols`)
  - derived semantic means:
    `:direction`, `:volatility`, `:vol_of_vol`, `:trend`, `:liquidity`
- `story_of_means_df`: A higher-level, human-readable or categorical
  interpretation of the semantic means, as produced by `story_of_means`.

# Method
1. Compute per-cluster means for all raw indicator features.
2. Aggregate indicator subsets into semantic dimensions using `mean_of_means`:
   - Direction (returns and momentum)
   - Volatility (realized and range-based measures)
   - Volatility of volatility
   - Trend (moving-average gaps and slopes)
   - Liquidity (volume and price-impact proxies)
3. Generate a semantic narrative or categorical summary via `story_of_means`.

# Notes
- All semantic dimensions are computed as simple arithmetic means of their
  constituent indicators.
- The function assumes all required indicator columns are present and numeric.
- Semantic aggregates are intended for interpretation and regime comparison,
  not model fitting.
  """
function calculate_cluster_means(clusters,feature_cols)

    table_of_means = indicator_means_df(clusters,feature_cols)

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

    table_of_means = mean_of_means(:direction,direction_cols,table_of_means)

    volatility_cols = [
        :realVol5day,
        :realVol10day,
        :realVol21day,
        :gkVol21day
    ]

    table_of_means = mean_of_means(:volatility,volatility_cols,table_of_means)

    vol_of_vol_cols = [
        :volOfVol21day
    ]

    table_of_means = mean_of_means(:vol_of_vol,vol_of_vol_cols,table_of_means)


    trend_cols = [
        :maDiff10_50day,
        :maDiff20_100day,
        :stSlope3day,
        :stSlope10day,
        :stSlope21day,
        :ema5MinusEma21,
        :ema21MinusEma100,
        :ema5daySlope,
        :ema10daySlope,
        :ema21daySlope
    ]

    table_of_means = mean_of_means(:trend,trend_cols,table_of_means)

    liquidity_cols = [
        :volumeZscore21day,
        :amihud21day,
        :lntrades
    ]

    table_of_means = mean_of_means(:liquidity,liquidity_cols,table_of_means)

    story_of_means_df = story_of_means(table_of_means)

    return table_of_means, story_of_means_df

    #return table_of_means

end


"""
    mean_of_means(col_name, cols, df)

Compute a row-wise mean over selected columns and store the result as a new column.

This function takes a set of numeric columns from a DataFrame and computes,
for each row, the arithmetic mean across those columns. The resulting value
is written into a new column named `col_name`. The operation is performed
in-place on the provided DataFrame.

### Arguments
- `col_name`: `Symbol` specifying the name of the output column to be added.
- `cols`: Vector of `Symbol`s identifying the columns whose row-wise mean
  should be computed.
- `df`: `DataFrame` containing the input columns.

### Returns
- `df`: The same `DataFrame` with an additional column `col_name` holding
  the row-wise means.
"""
function mean_of_means(col_name,cols,df)
        
    df[!,col_name]=mean.(eachrow(df[!,cols]))

    return df

end

"""
    story_of_means(table_of_means::DataFrame) -> DataFrame

Convert continuous semantic cluster means into a symbolic, interpretable representation.

For each cluster `k`, this function maps the mean values of the following
semantic dimensions:

- `:direction`
- `:volatility`
- `:vol_of_vol`
- `:trend`
- `:liquidity`

into qualitative signs based on fixed thresholds:

- `"+"` if value ≥ 0.5
- `"-"` if value ≤ -0.5
- `"0"` otherwise

The output is a DataFrame where each row corresponds to a cluster and each
semantic dimension is represented by a categorical sign in `{"+", "0", "-"}`.

This symbolic "story of means" is designed for human interpretability and
enables downstream operations such as semantic equivalence detection and
cluster collapsing.

### Arguments
- `table_of_means::DataFrame`: Must contain column `:k` and the five semantic
  mean columns listed above.

### Returns
- `DataFrame`: A semantic signature table with one row per cluster.
"""
function story_of_means(table_of_means)

    means_of_means_cols=[:direction,:volatility, :vol_of_vol, :trend, :liquidity]

    story_of_means_df=DataFrame(means_of_means_cols .=> Ref(String[]))
    
    insertcols!(story_of_means_df, 1, :k => Int64[])

    for _row in eachrow(table_of_means)

        new_row=[]

        push!(new_row,_row[:k])

        for _col in means_of_means_cols

            _sign = "0"

            _sign = _row[_col] <= -0.5 ? "-" : _sign

            _sign = _row[_col] >= 0.5 ? "+" : _sign

            push!(new_row,_sign)
            
        end

        push!(story_of_means_df,new_row)

    end

    story_of_means_df

end

"""
    sort_higherFreq_into_clusters(clusters, OHLCVT60_df)

Assign higher-frequency time-series data to existing low-frequency clusters
using rolling time windows.

For each cluster, this function takes the timestamps associated with the
cluster’s low-frequency observations and collects all higher-frequency
rows whose timestamps fall within the corresponding 24-hour window
`[timestamp, timestamp + 24h)`. The selected rows are appended and aggregated
per cluster.

This is typically used to project daily (or lower-frequency) clustering
results onto intraday (e.g. 60-minute) OHLCVT data.

### Arguments
- `clusters`: Dictionary mapping cluster identifiers to dictionaries that
  contain at least a `"data"` `DataFrame` with a `:timestamp` column.
- `OHLCVT60_df`: `DataFrame` of higher-frequency OHLCVT data containing
  a `:timestamp` column expressed in the same time units.

### Returns
- `OHLCVT60_clusters`: `Dict{Any,DataFrame}` where each key is a cluster
  identifier and each value is a `DataFrame` containing all higher-frequency
  observations assigned to that cluster.
"""
function sort_higherFreq_into_clusters(clusters, OHLCVT60_df)
    
    OHLCVT60_clusters = Dict{Any, DataFrame}()

    for (cluster_k, cluster_v) in clusters
       
        OHLCVT60_clusters[cluster_k] = DataFrame()

        for opening_bell in cluster_v["data"][!, :timestamp]
       
            closing_bell = opening_bell + 24*60*60

            cluster = filter(
                :timestamp => x -> (x >= opening_bell && x < closing_bell),
                OHLCVT60_df
            )

            append!(OHLCVT60_clusters[cluster_k], cluster)
       
        end
    
    end
    
    return OHLCVT60_clusters

end

"""
    data_into_clusters(OHLCVT_df, df_pca, labels)

Group original observations and their PCA representations into clusters.

This function builds a dictionary of clusters from a vector of cluster labels.
For each observation, the corresponding row from the PCA DataFrame and the
aligned row from the original OHLCVT DataFrame are inserted into the same
cluster entry. Each cluster stores its raw data and PCA-space data separately.

The mapping between `df_pca` and `OHLCVT_df` is established via the
`:row_idx` column in `df_pca`.

### Arguments
- `OHLCVT_df`: `DataFrame` containing the original OHLCVT observations.
- `df_pca`: `DataFrame` containing PCA features and a `:row_idx` column
  pointing to rows in `OHLCVT_df`.
- `labels`: Vector of cluster assignments, one per row of `df_pca`.

### Returns
- `cluster_numbers`: `Dict` mapping each cluster label to a dictionary with:
  - `"data"`: `DataFrame` of OHLCVT rows belonging to the cluster.
  - `"pca"`: `DataFrame` of PCA rows belonging to the cluster.
"""
function data_into_clusters(OHLCVT_df, df_pca, labels)
    
    cluster_numbers=Dict()
    dict_keys=unique(labels)
    for v in dict_keys
        if !haskey(cluster_numbers,v)
            push!(cluster_numbers,v=>Dict("data"=>DataFrame(),"pca"=>DataFrame()))
        end 
    end

    for (i,v) in enumerate(labels)
        push!(cluster_numbers[v]["pca"],df_pca[i, :])
        push!(cluster_numbers[v]["data"],OHLCVT_df[df_pca[!,:row_idx][i], :])
    end

    return cluster_numbers

end

"""
    add_cluster_labels(
        collapsed_clusters::Dict,
        OHLCVT_df::DataFrame
    ) -> DataFrame

Attach cluster (regime) labels to a time-indexed OHLCVT DataFrame.

This function constructs a lookup table from timestamps to cluster identifiers
using the `"data"` DataFrame stored in each entry of `collapsed_clusters`.
Each timestamp in `OHLCVT_df` is then assigned the corresponding cluster label.
Timestamps that do not belong to any cluster are assigned `missing`.

The function mutates and returns `OHLCVT_df` by adding a new column `:labels`.

# Arguments
- `collapsed_clusters::Dict`: Dictionary mapping cluster IDs to cluster payloads.
  Each payload must contain a `"data"` entry with a `:timestamp` column.
- `OHLCVT_df::DataFrame`: Time-indexed OHLCVT DataFrame containing a `:timestamp` column.

# Returns
- `DataFrame`: The input `OHLCVT_df` with an added `:labels` column containing
  cluster identifiers (`Int64`) or `missing` where no cluster assignment exists.

# Notes
- If a timestamp appears in multiple clusters, the last assignment encountered
  in iteration order will prevail.
- No sorting or validation of timestamps is performed.
- The function assumes timestamps are comparable using equality (`==`).

"""
function add_cluster_labels(collapsed_clusters::Dict, OHLCVT_df::DataFrame)::DataFrame
    # Build a timestamp -> cluster_number lookup from all clusters
    label_map = Dict{Int64, Union{Int64, Missing}}()
    for (cluster_id, cluster_dict) in collapsed_clusters
        cluster_df=cluster_dict["data"]
        for ts in cluster_df.timestamp
            label_map[ts] = cluster_id
        end
    end

    # Map each timestamp in OHLCVT_df to its label, defaulting to missing
    labels = [get(label_map, ts, missing) for ts in OHLCVT_df.timestamp]

    # Return a copy with the new column
    OHLCVT_df.labels = labels
    return OHLCVT_df
end

"""
    regime_mean_nextday_return(
        df::DataFrame;
        regime_col::Symbol = :labels,
        return_col::Symbol = :NextDayReturn
    ) -> Dict{Int, Float64}

Compute the mean next-day return conditional on market regime.

For each regime `k`, this function constructs the conditional return set

    Rₖ = { rₜ₊₁ | Rₜ = k }

using the column `return_col`, and computes the conditional mean

    μₖ = E[rₜ₊₁ | Rₜ = k].

Rows with missing regime labels or missing returns are excluded from the
calculation.

# Arguments
- `df::DataFrame`: Input DataFrame containing regime labels and next-day returns.
- `regime_col::Symbol`: Column name identifying the regime label at time `t`
  (default: `:labels`).
- `return_col::Symbol`: Column name containing the next-day return `rₜ₊₁`
  (default: `:NextDayReturn`).

# Returns
- `Dict{Int, Float64}`: Dictionary mapping each regime label `k` to its
  mean next-day return `μₖ`.

# Notes
- The function assumes regime labels are integer-valued.
- Returns are interpreted as raw or log returns depending on how
  `return_col` was constructed.
- No adjustment for transaction costs or overlapping returns is performed.

"""

function regime_mean_nextday_return(df::DataFrame;
    regime_col::Symbol = :labels,
    return_col::Symbol = :NextDayReturn
)
    mask = .!ismissing.(df[!, regime_col]) .& .!ismissing.(df[!, return_col])
    df_clean = df[mask, :]

    result = Dict{Int, Float64}()

    for k in unique(df_clean[!, regime_col])
        Rk = df_clean[df_clean[!, regime_col] .== k, return_col]
        result[k] = mean(Rk)
    end

    return result
end

"""
    regime_nextday_volatility(
        df::DataFrame;
        regime_col::Symbol = :labels,
        return_col::Symbol = :NextDayReturn
    ) -> Dict{Int, Float64}

Compute the volatility of next-day returns conditional on market regime.

For each regime `k`, this function forms the conditional return set

    Rₖ = { rₜ₊₁ | Rₜ = k }

and computes the sample standard deviation

    σₖ = sqrt(Var[rₜ₊₁ | Rₜ = k]).

Rows with missing regime labels or missing returns are excluded from the
calculation.

# Arguments
- `df::DataFrame`: Input DataFrame containing regime labels and next-day returns.
- `regime_col::Symbol`: Column identifying the regime at time `t`
  (default: `:labels`).
- `return_col::Symbol`: Column containing the next-day return `rₜ₊₁`
  (default: `:NextDayReturn`).

# Returns
- `Dict{Int, Float64}`: Dictionary mapping each regime label `k` to its
  conditional next-day return volatility `σₖ`.

# Notes
- Volatility is computed using the sample standard deviation (`n − 1` normalization).
- Returns may be raw or log returns depending on how `return_col` is defined.
- This metric captures *risk conditional on regime*, not intraregime realized volatility.

"""
function regime_nextday_volatility(
    df::DataFrame;
    regime_col::Symbol = :labels,
    return_col::Symbol = :NextDayReturn
)
    mask = .!ismissing.(df[!, regime_col]) .& .!ismissing.(df[!, return_col])
    df_clean = df[mask, :]

    result = Dict{Int, Float64}()

    for k in unique(df_clean[!, regime_col])
        Rk = df_clean[df_clean[!, regime_col] .== k, return_col]
        result[k] = std(Rk)
    end

    return result
end

"""
    regime_nextday_downside_deviation(
        df::DataFrame;
        regime_col::Symbol = :labels,
        return_col::Symbol = :NextDayReturn
    ) -> Dict{Int, Float64}

Compute the downside deviation of next-day returns conditional on market regime.

For each regime `k`, this function constructs the conditional return set

    Rₖ = { rₜ₊₁ | Rₜ = k }

and the subset of negative returns

    Rₖ⁻ = { r ∈ Rₖ | r < 0 }.

The downside deviation is then computed as

    σₖ⁻ = sqrt( (1 / |Rₖ⁻|) * ∑ r² ),   r ∈ Rₖ⁻,

which corresponds to the standard semi-deviation with a zero target return.

Rows with missing regime labels or missing returns are excluded from the
calculation.

# Arguments
- `df::DataFrame`: Input DataFrame containing regime labels and next-day returns.
- `regime_col::Symbol`: Column identifying the regime at time `t`
  (default: `:labels`).
- `return_col::Symbol`: Column containing the next-day return `rₜ₊₁`
  (default: `:NextDayReturn`).

# Returns
- `Dict{Int, Float64}`: Dictionary mapping each regime label `k` to its
  conditional downside deviation `σₖ⁻`.

# Notes
- No mean subtraction is performed; the target return is zero.
- If a regime has no negative next-day returns, its downside deviation is
  returned as `0.0`.
- This metric is commonly used in downside-risk and Sortino ratio calculations.
"""
function regime_nextday_downside_deviation(
    df::DataFrame;
    regime_col::Symbol = :labels,
    return_col::Symbol = :NextDayReturn
)
    mask = .!ismissing.(df[!, regime_col]) .& .!ismissing.(df[!, return_col])
    df_clean = df[mask, :]

    result = Dict{Int, Float64}()

    for k in unique(df_clean[!, regime_col])
        Rk = df_clean[df_clean[!, regime_col] .== k, return_col]

        Rk_neg = Rk[Rk .< 0]

        if isempty(Rk_neg)
            result[k] = 0.0
        else
            result[k] = sqrt(mean(Rk_neg .^ 2))
        end
    end

    return result
end

function print_regime_nextday_summary(
    mean_ret::Dict{Int, Float64},
    vol::Dict{Int, Float64},
    downside::Dict{Int, Float64};
    digits::Int = 4
)
    regimes = sort(collect(union(keys(mean_ret), keys(vol), keys(downside))))

    println("Next-Day Regime Statistics")
    println("--------------------------")
    println(
        rpad("Regime", 10),
        rpad("Mean", 15),
        rpad("Volatility", 15),
        "Downside Dev"
    )

    for k in regimes
        μ  = get(mean_ret, k, NaN)
        σ  = get(vol, k, NaN)
        σd = get(downside, k, NaN)

        println(
            rpad(string(k), 10),
            rpad(round(μ, digits=digits), 15),
            rpad(round(σ, digits=digits), 15),
            round(σd, digits=digits)
        )
    end
end

function check_clusters_predictive_quality(collapsed_cluster, OHLCVT_df)
    #Add the daily labels from collapsed_cluster as an extra column to OHLCVT_df 
    OHLCVT_df=add_cluster_labels(collapsed_cluster, OHLCVT_df)
    
    #Calculate next day return for each timestep
    lnReturn1day=logreturn(OHLCVT_df[!,:close], 1)
    popfirst!(lnReturn1day)
    OHLCVT_df.NextDayReturn=push!(lnReturn1day,missing)
    
    #=Create sets Rk​={rt+1​∣Rt​=k} and calculate their mean: μk​=1/∣Rk​∣ ​∑_(​r∈Rk)​r
        Interpretation:
        Positive → regime is favorable on average
        Negative → regime is hostile
        Near zero → no directional edge
        =#
    regime_nd_mean=regime_mean_nextday_return(OHLCVT_df)
    
    #=Calculate the volatility: σk​=sqrt(1/(∣Rk​∣−1) ​∑_​(r∈Rk)​(r−μk​)^2)
​        Interpretation:
        High volatility → unstable / risky regime
        Low volatility → calm regime
        =#
    regime_nd_volatility=regime_nextday_volatility(OHLCVT_df)
    
    #=Calculate Downside deviation: σk−​=sqrt(1/∣Rk−​∣ ∑_(​r∈Rk-)​r^2) where Rk− is days where a loss occured.
        Interpretation:
        Large value → painful losses
        Small value → shallow drawdowns
        =#
    regime_nd_downside_deviation=regime_nextday_downside_deviation(OHLCVT_df)
    #=One wants to see clear separation, e.g.:
    Regime A: low vol, small positive mean
    Regime B: high vol, negative mean
    Regime C: medium vol, fat downside
    If everything looks similar → clustering failed.
    =#
    print_regime_nextday_summary(
        regime_nd_mean,
        regime_nd_volatility,
        regime_nd_downside_deviation
    )
    return Dict("means"=>regime_nd_mean,
        "volatility"=>regime_nd_volatility,
        "downside"=>regime_nd_downside_deviation)

end