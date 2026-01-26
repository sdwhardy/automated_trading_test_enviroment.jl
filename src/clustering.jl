"""
loglikelihood(gmm::GMM, X::Matrix{Float64}) -> Float64

Compute the total log-likelihood of a dataset under a fitted Gaussian Mixture Model (GMM).

Each sample's likelihood is first evaluated under the mixture model using `llpg`,
which returns the likelihood per data point. The natural logarithm is then applied
element-wise, and the results are summed over all samples.

# Arguments
- `gmm::GMM`: A fitted Gaussian mixture model from `GaussianMixtures.jl`.
- `X::Matrix{Float64}`: Data matrix of size `(N, d)`, where rows are samples and
  columns are features.

# Returns
- `Float64`: The total log-likelihood of the dataset under the model.

# Notes
- This function assumes `llpg(gmm, X)` returns strictly positive likelihood values.
- The result scales linearly with the number of samples and is not normalized.

"""
function loglikelihood(gmm::GMM, X::Matrix{Float64})
    return sum(log.(llpg(gmm, X)))
end

"""
    gmm_bic(gmm::GMM, X::Matrix)

Compute the Bayesian Information Criterion (BIC) for a fitted Gaussian Mixture
Model with diagonal covariance matrices.

The BIC is defined as

    BIC = -2 * log L + p * log(n)

where:
- `log L` is the total log-likelihood of the data under the model,
- `p` is the number of free parameters,
- `n` is the number of observations.

For a diagonal-covariance GMM with `k` components and `d` dimensions:
- means contribute `k * d` parameters,
- variances contribute `k * d` parameters,
- mixture weights contribute `k - 1` parameters.

The total log-likelihood is reconstructed from the average log-likelihood
returned by `avll(gmm, X)`.

# Arguments
- `gmm::GMM`: A fitted GaussianMixtures.jl GMM object.
- `X::Matrix`: Data matrix of size `(n, d)` with rows as samples.

# Returns
- `Float64`: Bayesian Information Criterion value (lower is better).
"""
function gmm_bic(gmm::GMM, X::Matrix)
    n, d = size(X)

    avg_ll = avll(gmm, X)
    ll = n * avg_ll   # IMPORTANT

    k = gmm.n
    p = k * (2d) + (k - 1)   # diag covariance

    return -2 * ll + p * log(n)
end
"""
    select_k_bic(X::Matrix; ks=2:15, kind=:diag)

Selects the optimal number of Gaussian mixture components using the
Bayesian Information Criterion (BIC).

For each candidate number of clusters `k` in `ks`, a Gaussian Mixture Model
(GMM) is fitted to the data `X` using the EM algorithm. The BIC score is
computed for each fitted model, and the model with the minimum BIC is selected.

If model fitting fails for a given `k`, the corresponding BIC is set to `Inf`
and the model is skipped.

# Arguments
- `X::Matrix`: Data matrix with rows as samples and columns as features.
- `ks`: Range or collection of candidate cluster counts (default: `2:15`).
- `kind::Symbol`: Covariance type (`:diag` or `:full`, default: `:diag`).

# Returns
- `k_opt`: Optimal number of clusters.
- `gmm_opt`: Fitted GMM corresponding to `k_opt`.
- `bics`: Vector of BIC values for all tested `k`.

# Notes
- BIC is defined as `BIC = -2 log L + p log n`
- Lower BIC indicates a better trade-off between fit quality and model complexity.
"""
function select_k_bic(X::Matrix; ks=2:15, kind=:diag)
    bics = Float64[]
    models = GMM[]

    for k in ks
        try
            #gmm = GMM(k, X; kind=kind, nInit=10, nIter=50)
            gmm = GMM(k,X;kind   = :diag,
    method = :kmeans,
    nIter  = 150,
    nInit  = 1,
    nFinal = 10,
    parallel = false)
            em!(gmm, X; nIter=50)
            if any(isnan, gmm.Σ) || any(x -> x < 1e-8, gmm.Σ)
                warning("Degenerate covariance")
            else
                push!(bics, gmm_bic(gmm, X))
                println(k," bic is: ",bics[end])
                push!(models, gmm)
            end
        catch err
            @warn "GMM failed for k=$k" err
            push!(bics, Inf)
        end
    end

    kopt_idx = argmin(bics)
    return ks[kopt_idx], models[kopt_idx], bics
end

"""
    num_params_full(k, d)

Compute the number of free parameters in a Gaussian Mixture Model (GMM)
with **full covariance matrices**.

# Parameters
- `k::Int`: Number of mixture components.
- `d::Int`: Data dimensionality (number of features).

# Returns
- `Int`: Total number of independent model parameters.

# Details
The parameter count is composed of:
- Mixture weights: `k - 1` (due to the sum-to-one constraint)
- Means: `k * d`
- Covariances: `k * d * (d + 1) / 2` (symmetric full covariance matrices)

This count is typically used when computing information criteria such as
BIC or AIC.

# Formula
Total parameters:
(k - 1) + k d + k d (d + 1) / 2
"""
function num_params_full(k, d)
    # weights (k-1) + means (k*d) + covariances (k*d*(d+1)/2)
    return (k - 1) + k*d + k*d*(d + 1) ÷ 2
end

"""
mean_entropy(post)

Compute the mean posterior entropy of a Gaussian Mixture Model clustering.

Each row of `post` corresponds to the posterior cluster probabilities
for a single sample. The entropy is computed per sample and then averaged
over all samples.

This metric is commonly used as a soft-clustering validation measure:
lower values indicate more confident (sharper) cluster assignments.

# Arguments
- `post::AbstractMatrix{<:Real}`: Posterior probability matrix of size (N × K),
  where N is the number of samples and K the number of clusters.

# Returns
- `Float64`: Mean entropy across all samples.
"""
function mean_entropy(post)
    return mean(-sum(p .* log.(p .+ eps())) for p in eachrow(post))
end

function bootstrap_stability(X, k, ref_labels; B=50)
    
    scores = Float64[]
    for i in 1:B
        idx = sample(1:size(X,1), size(X,1), replace=true)
        Xb = X[idx, :]

        gmm = GMM(k, Xb; kind=:diag, nInit=10, nIter=50)
        em!(gmm, Xb; nIter=200)
        if any(isnan, gmm.Σ) || any(x -> x < 1e-8, gmm.Σ)
            warning("Degenerate covariance")
        else
            post, _ = gmmposterior(gmm, X)
            labels = argmax.(eachrow(post))
            push!(scores, randindex(ref_labels, labels)[2])
        end
    end
    
    return mean(scores), std(scores)

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

function indicator_means_df(clusters,feature_cols)
    table_of_means=DataFrame(feature_cols .=> Ref(Float64[]))
    insertcols!(table_of_means, 1, :k => Int64[])
    for k in keys(clusters)
        cols_means=[]
        push!(cols_means,k)
        for col in feature_cols

            push!(cols_means,mean(clusters[k]["data"][!,col]))

        end

        push!(table_of_means,cols_means)
    end
    return table_of_means
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




function k_gmm_clusters(df_pca,k)

    X = Matrix(df_pca[:, Not(:row_idx)])

    # Gaussian mixture model clustering 
    best_k, best_gmm, best_bics = select_k_bic(X; ks=k:k, kind=:diag)

    post, _ = gmmposterior(best_gmm, X)

    best_labels = argmax.(eachrow(post))

    entropy = mean_entropy(post)# must be <0.3

    m,s = bootstrap_stability(X, best_k, best_labels; B=50)

    println("Entropy: ", entropy)

    for _k = 1:1:best_k
        println(_k,": ", count(x -> x == _k, best_labels))#669 
    end

    println("Stability Mean: ", m, ", Std: ",s)

    return Dict("best_k"=>best_k, "best_labels"=>best_labels, "m"=>m, "s"=>s, "entropy"=>entropy, "bic"=>best_bics[argmin(best_bics)])

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

function select_best_gmm(gmm_results::Dict{Any,Any};
    entropy_weight::Float64 = 100.0,
    max_entropy::Float64 = 0.35
)
    best_score = Inf
    best_K = nothing
    best_iter = nothing

    for (K, iter_dict) in gmm_results
        for (iter, payload) in iter_dict
            clusters = payload["gmm_clusters"]

            bic = clusters["bic"]
            entropy = clusters["entropy"]

            # basic validity checks
            if !isfinite(bic) || !isfinite(entropy)
                continue
            end
            if entropy > max_entropy
                continue
            end

            score = bic + entropy_weight * entropy

            if score < best_score
                best_score = score
                best_K = K
                best_iter = iter
            end
        end
    end

    best_K === nothing && error("No valid GMM configuration found")

    return (
        cluster_number = best_K,
        iteration = best_iter,
        score = best_score
    )
end


"""
Collapse statistically indistinguishable GMM clusters.

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
Collapse GMM clusters that are semantically equivalent.

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
    k_clusters_i_times(df_pca, OHLCVT_df, feature_cols; _k=5, _i=5)

Run Gaussian Mixture Model (GMM) clustering multiple times for a range of cluster
counts and collect detailed results for each run.

For each number of clusters `k = 2:_k`, the GMM clustering is executed `_i` times.
For every run, the function:
1. Computes GMM clusters on the PCA-transformed data.
2. Maps the resulting cluster labels back to the original OHLCVT data.
3. Computes per-cluster feature means and summary statistics.
4. Stores all intermediate and final results in a nested dictionary.

# Arguments
- `df_pca`: DataFrame containing PCA-transformed features used for clustering.
- `OHLCVT_df`: Original DataFrame containing OHLCVT data to be grouped by cluster.
- `feature_cols`: Vector of column names used when computing cluster means.

# Keyword Arguments
- `_k` (default = 5): Maximum number of clusters to evaluate (minimum is fixed at 2).
- `_i` (default = 5): Number of repeated GMM runs per cluster count.

# Returns
- `Dict`: A nested dictionary structured as:
  `results[k][i]`, where each entry contains:
    - `"gmm_clusters"`: Raw GMM clustering output.
    - `"clusters"`: OHLCVT data split by cluster.
    - `"table_of_means"`: Table of per-cluster feature means.
    - `"story_of_means_df"`: DataFrame with descriptive statistics per cluster.
"""

function k_clusters_i_times(
    df_pca,
    OHLCVT_df,
    feature_cols;
    _k::Int = 5,
    _i::Int = 5,
)
    # Top-level container: keyed by number of clusters k
    gmm_results = Dict()

    # Iterate over number of clusters
    for k in 2:_k
        push!(gmm_results, k => Dict())

        # Repeat clustering multiple times for robustness
        for i in 1:_i
            push!(gmm_results[k], i => Dict())

            # Perform GMM clustering in PCA space
            gmm_clusters = k_gmm_clusters(df_pca, k)

            # Map cluster labels back to the original OHLCVT data
            clusters = data_into_clusters(
                OHLCVT_df,
                df_pca,
                gmm_clusters["best_labels"],
            )

            # Compute per-cluster feature statistics
            table_of_means, story_of_means_df =
                calculate_cluster_means(clusters, feature_cols)

            # Store results for this (k, i) combination
            push!(gmm_results[k][i], "gmm_clusters" => gmm_clusters)
            push!(gmm_results[k][i], "clusters" => clusters)
            push!(gmm_results[k][i], "table_of_means" => table_of_means)
            push!(gmm_results[k][i], "story_of_means_df" => story_of_means_df)
        end
    end

    return gmm_results
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




function mean_silhouette(X, labels)

    S = silhouettes(labels, X; metric = SqEuclidean())

    return mean(S)

end

function clustering_stability(X, k; trials=8)

    label_sets = Vector{Vector{Int}}()

    for _ in 1:trials

        R = kmeans(X, k; maxiter=300, init=:rand)

        push!(label_sets, R.assignments)

    end


    agreements = Float64[]

    for i in 1:length(label_sets), j in i+1:length(label_sets)

        push!(agreements, mean(label_sets[i] .== label_sets[j]))

    end

    return mean(agreements)

end

function gmm_cluster_set(df_pca,OHLCVT_df, pca_cols)
            
    gmm_results=k_clusters_i_times(df_pca,OHLCVT_df, pca_cols;_k=5, _i=5)

    best = select_best_gmm(gmm_results)
    
    return gmm_results[best.cluster_number][best.iteration] 

end

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


function best_k4k_means(df_pca; n_restarts = 8, stability_thresh = 0.85, min_sil_gain = 0.03) 
    # Extract PCA features only (drop row_idx)
    X = Matrix(df_pca[:, Not(:row_idx)])'   # 5650 × 8

    # Clustering.jl expects features × samples
    n_points = size(X, 2)         # number of samples

    ks = 2:min(10, n_points)      # ensure k never exceeds N # hard cap per your use case

    results = Dict{Int,Tuple{Float64,Float64}}()

    for k in ks
        R = kmeans(X, k; maxiter=300, init=:rand)
        sil = mean_silhouette(X, R.assignments)
        stab = clustering_stability(X, k; trials=n_restarts)
        results[k] = (sil, stab)
    end

    # Filter by stability
    valid = filter(k -> results[k][2] ≥ stability_thresh, ks)

    if isempty(valid)
        # fallback: smallest k with max stability
        best_k = argmax(k -> results[k][2], ks)
    else
        # choose smallest k with max silhouette among stable ones
        best_k = argmax(k -> results[k][1], valid)
    end

    if results[best_k][1] < min_sil_gain
        best_k = 1
    end

    return best_k
    
end


function k_means_set(df_pca, best_k, OHLCVT)

    # Extract PCA features only (drop row_idx)
    X = Matrix(df_pca[:, Not(:row_idx)])'   # 5650 × 8

    #k means is working but untested and still no validation of picking best clusters (eg is 1 better than the 2 found?)
    R = kmeans(X, best_k; maxiter=300, init=:rand)

    kmeans_cluster=data_into_clusters(OHLCVT, df_pca, R.assignments)

    return kmeans_cluster

end