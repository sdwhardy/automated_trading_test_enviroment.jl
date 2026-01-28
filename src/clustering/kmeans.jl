

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

"""
    k_means_set(df_pca, best_k, OHLCVT)

Apply k-means clustering to PCA-transformed data and map cluster assignments
back to the original OHLCVT dataset.

This function performs k-means clustering on PCA features extracted from
`df_pca`, assigns each observation to one of `best_k` clusters, and aggregates
the original OHLCVT data according to these cluster assignments.

# Arguments
- `df_pca`: `DataFrame` containing PCA-transformed features. All columns except
  `:row_idx` are treated as numeric inputs to the clustering.
- `best_k`: Number of clusters to form.
- `OHLCVT`: `DataFrame` containing the original time-series or market data
  (e.g. Open, High, Low, Close, Volume, Time) to be grouped by cluster.

# Returns
- A clustering payload produced by `data_into_clusters`, containing the
  OHLCVT data partitioned according to k-means assignments.

# Method
1. Extract PCA feature matrix from `df_pca` and transpose it to match the
   expected input shape of `kmeans`.
2. Run k-means clustering with random initialization and a maximum of
   300 iterations.
3. Assign each observation to its nearest centroid.
4. Aggregate the OHLCVT data using the resulting cluster assignments.

# Notes
- Cluster initialization is random (`init = :rand`); results may vary between
  runs unless a random seed is fixed externally.
- No internal validation (e.g. silhouette score, inertia comparison across
  `k`) is performed to assess whether `best_k` is optimal.
- PCA features are assumed to be scaled appropriately prior to clustering.

"""
function k_means_set(df_pca, best_k, OHLCVT)

    # Extract PCA features only (drop row_idx)
    X = Matrix(df_pca[:, Not(:row_idx)])'   # 5650 × 8

    #k means is working but untested and still no validation of picking best clusters (eg is 1 better than the 2 found?)
    R = kmeans(X, best_k; maxiter=300, init=:rand)

    kmeans_cluster=data_into_clusters(OHLCVT, df_pca, R.assignments)

    return kmeans_cluster

end

"""
    mean_silhouette(X, labels) -> Float64

Compute the mean silhouette coefficient for a clustering assignment.

This function evaluates clustering quality by computing the silhouette
coefficient for each observation and returning their average. The silhouette
measures how well each point fits within its assigned cluster compared to
neighboring clusters.

# Arguments
- `X`: Matrix of size `(n_features, n_samples)` containing the feature vectors
  used for clustering.
- `labels`: Vector of cluster assignments of length `n_samples`.

# Returns
- `Float64`: Mean silhouette coefficient across all observations.

# Method
- Pairwise distances are computed using squared Euclidean distance.
- The silhouette coefficient is computed for each observation.
- The mean silhouette score is returned.

# Interpretation
- Values close to `1` indicate well-separated, compact clusters.
- Values near `0` indicate overlapping clusters.
- Negative values indicate likely misclassification.

# Notes
- This function assumes a hard clustering assignment.
- The squared Euclidean distance preserves ordering of distances and is
  consistent with k-means–style objectives.
- No validation is performed on label cardinality or cluster sizes.

"""
function mean_silhouette(X, labels)

    S = silhouettes(labels, X; metric = SqEuclidean())

    return mean(S)

end

"""
    clustering_stability(X, k; trials=8) -> Float64

Estimate k-means clustering stability via repeated random initializations.

This function evaluates the stability of a k-means clustering by running the
algorithm multiple times with random initialization and measuring the average
pairwise agreement between the resulting cluster assignments.

# Arguments
- `X`: Matrix of size `(n_features, n_samples)` containing the data used for
  clustering.
- `k`: Number of clusters.
- `trials`: Number of independent k-means runs with random initialization
  (default: `8`).

# Returns
- `Float64`: Mean pairwise label agreement across all trial combinations.

# Method
1. Run k-means `trials` times with random initialization.
2. Collect cluster assignment vectors from each run.
3. Compute pairwise agreement between all distinct pairs of labelings.
4. Return the mean agreement score.

# Interpretation
- Values close to `1` indicate highly stable cluster assignments.
- Lower values indicate sensitivity to initialization and potential ambiguity
  in cluster structure.

# Notes
- Agreement is measured as the fraction of observations assigned to the same
  cluster index across runs.
- This metric is sensitive to label permutation; cluster relabeling is not
  accounted for.
- Intended as a heuristic stability measure rather than a formal statistical
  test.

"""
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