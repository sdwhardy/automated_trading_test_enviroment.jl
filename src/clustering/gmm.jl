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

"""
    bootstrap_stability(X, k, ref_labels; B=50) -> (mean_score, std_score)

Estimate clustering stability using bootstrap resampling and the Rand index.

This function evaluates how stable a Gaussian Mixture Model (GMM) clustering
with `k` components is under bootstrap resampling of the data. For each bootstrap
replicate, a GMM is fitted to a resampled dataset and the resulting clustering
is compared to a reference labeling using the Rand index.

# Arguments
- `X`: A matrix of size `(n_samples, n_features)` containing the input data.
- `k`: Number of mixture components (clusters) for the GMM.
- `ref_labels`: Vector of reference cluster labels of length `n_samples`,
  used as the baseline for stability comparison.
- `B`: Number of bootstrap replicates (default: `50`).

# Returns
- `mean_score::Float64`: Mean Rand index across valid bootstrap runs.
- `std_score::Float64`: Standard deviation of the Rand index across runs.

# Method
For each bootstrap iteration:
1. Rows of `X` are resampled with replacement.
2. A diagonal-covariance GMM is fitted using EM.
3. Posterior cluster probabilities are evaluated on the *original* dataset `X`.
4. Hard labels are assigned via maximum posterior probability.
5. The Rand index is computed against `ref_labels`.

Bootstrap runs producing degenerate covariance matrices
(`NaN` values or variances below `1e-8`) are discarded.

# Notes
- Stability is measured relative to the provided reference labeling, not
  pairwise between bootstrap runs.
- Missing or ill-conditioned covariance estimates reduce the effective
  sample size used in the final statistics.
- The Rand index reported corresponds to the *second* output of `randindex`,
  i.e. the adjusted or normalized variant, depending on implementation.

"""
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

"""
    k_gmm_clusters(df_pca, k) -> Dict{String, Any}

Fit a Gaussian Mixture Model (GMM) with `k` components to PCA-transformed data
and evaluate clustering quality and stability.

This function applies diagonal-covariance GMM clustering to a PCA feature
DataFrame, computes hard cluster assignments, and reports multiple quality
metrics including entropy and bootstrap stability.

# Arguments
- `df_pca`: A `DataFrame` containing PCA features. All columns except `:row_idx`
  are treated as numeric input features.
- `k`: Number of mixture components to fit. Internally passed as a fixed range
  `k:k` to BIC-based model selection.

# Returns
A `Dict{String, Any}` with the following keys:
- `"best_k"`: Selected number of clusters (equal to `k`).
- `"best_labels"`: Vector of hard cluster labels assigned to each observation.
- `"m"`: Mean Rand index from bootstrap stability analysis.
- `"s"`: Standard deviation of the Rand index from bootstrap stability analysis.
- `"entropy"`: Mean posterior entropy across observations.
- `"bic"`: Bayesian Information Criterion (BIC) value of the fitted model.

# Method
1. Extract numeric PCA features from `df_pca`.
2. Fit a diagonal-covariance GMM using EM.
3. Select the model via BIC (degenerate here since `k` is fixed).
4. Compute posterior probabilities and assign hard labels.
5. Evaluate clustering entropy as a measure of assignment confidence.
6. Estimate clustering stability via bootstrap resampling.

# Side Effects
- Prints entropy, per-cluster sample counts, and bootstrap stability statistics
  to standard output.

# Notes
- The entropy threshold (`< 0.3`) is not enforced programmatically and is
  provided as a guideline only.
- The function assumes `df_pca` contains no missing values in feature columns.
- Stability is measured relative to the clustering obtained on the full dataset.

"""
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

"""
    select_best_gmm(gmm_results; entropy_weight=100.0, max_entropy=0.35)
        -> (cluster_number, iteration, score)

Select the best Gaussian Mixture Model (GMM) configuration based on a
BIC–entropy trade-off.

This function scans a nested results dictionary containing multiple GMM
configurations and selects the model that minimizes a composite score
defined as:

score = BIC + entropy_weight * entropy


subject to an upper bound on acceptable clustering entropy.

# Arguments
- `gmm_results`: A nested dictionary indexed as
  `gmm_results[K][iter]`, where each leaf entry contains a dictionary
  under the key `"gmm_clusters"` with at least:
  - `"bic"`: Bayesian Information Criterion value (`Float64`)
  - `"entropy"`: Mean posterior entropy (`Float64`)
- `entropy_weight`: Weight applied to entropy in the composite score
  (default: `100.0`).
- `max_entropy`: Maximum allowed entropy for a configuration to be
  considered valid (default: `0.35`).

# Returns
A named tuple with fields:
- `cluster_number`: Selected number of clusters `K`.
- `iteration`: Iteration index corresponding to the selected model.
- `score`: Composite score of the selected configuration.

# Selection Rules
- Configurations with non-finite BIC or entropy are discarded.
- Configurations with entropy greater than `max_entropy` are discarded.
- Among remaining candidates, the configuration minimizing the composite
  score is selected.

# Errors
- Throws an error if no valid GMM configuration satisfies the selection
  criteria.

# Notes
- The weighting scheme implicitly prioritizes BIC while penalizing
  uncertain cluster assignments via entropy.
- The choice of `entropy_weight` and `max_entropy` is application-specific
  and should be tuned empirically.
- This function assumes lower BIC and lower entropy are both preferable.

"""
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

"""
    gmm_cluster_set(df_pca, OHLCVT_df, pca_cols)

Compute multiple GMM clustering configurations and return the best-performing
cluster set based on BIC–entropy selection.

This function runs Gaussian Mixture Model (GMM) clustering for a fixed number
of clusters and repeated initializations, evaluates all configurations, and
returns the clustering payload corresponding to the best configuration as
determined by `select_best_gmm`.

# Arguments
- `df_pca`: `DataFrame` containing PCA-transformed features used for clustering.
- `OHLCVT_df`: `DataFrame` containing associated time-series or market data
  (e.g. Open, High, Low, Close, Volume, Time), passed through to downstream
  clustering routines.
- `pca_cols`: Vector of column names in `df_pca` identifying the PCA features
  to be used in clustering.

# Returns
- A dictionary (or payload object) corresponding to a single GMM configuration,
  as produced by `k_clusters_i_times`, containing clustering results and
  diagnostics (e.g. labels, entropy, BIC).

# Method
1. Generate GMM clustering results for a fixed cluster count (`_k = 5`)
   and multiple initializations (`_i = 5`).
2. Evaluate all configurations using `select_best_gmm`, which balances
   BIC and entropy.
3. Return the clustering result corresponding to the selected configuration.

# Notes
- The number of clusters and iterations are currently hard-coded
  (`_k = 5`, `_i = 5`).
- No validation is performed on the structure of the returned payload; it is
  assumed to be compatible with `select_best_gmm`.
- Any printing or logging behavior occurs in downstream functions.

"""
function gmm_cluster_set(df_pca,OHLCVT_df, pca_cols)
            
    gmm_results=k_clusters_i_times(df_pca,OHLCVT_df, pca_cols;_k=5, _i=5)

    best = select_best_gmm(gmm_results)
    
    return gmm_results[best.cluster_number][best.iteration] 

end

