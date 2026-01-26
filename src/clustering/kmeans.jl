

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