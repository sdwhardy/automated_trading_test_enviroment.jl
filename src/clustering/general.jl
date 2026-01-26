
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



function mean_of_means(col_name,cols,df)
        
    df[!,col_name]=mean.(eachrow(df[!,cols]))

    return df

end


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