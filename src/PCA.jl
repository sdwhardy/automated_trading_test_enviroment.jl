

function PCA(df::DataFrame, feature_cols::Vector{Symbol}; percent_explained::Float64=0.95)


    # ----------------------------
    # 1. Identify rows with no missing values in features
    # ----------------------------
    X_all = Matrix(df[:, feature_cols])
    row_mask = all(.!ismissing.(X_all), dims=2)   # true if row has no missing
    row_mask_vec = vec(row_mask)
    nonmissing_idx = findall(row_mask_vec)

    # Subset DataFrame and matrix
    df_nonmissing = df[nonmissing_idx, :]
    X = Matrix(df_nonmissing[:, feature_cols])

    # ----------------------------
    # 2. Center the data (features are already Z-scored, but centering is safe)
    # ----------------------------
    X_centered = X .- mean(X, dims=1)

    # ----------------------------
    # 3. Compute PCA via covariance and eigen decomposition
    # ----------------------------
    Σ = cov(X_centered, dims=1)
    eigvals, eigvecs = eigen(Σ)

    # Sort eigenvalues descending
    idx = sortperm(eigvals, rev=true)

    eigvals_sorted = eigvals[idx]
    eigvecs_sorted = eigvecs[:, idx]

    #  PC scores with sorted eigenvectors
    PC_scores_sorted = X_centered * eigvecs_sorted

    # ----------------------------
    # 4. Optional: Variance explained by each PC
    # ----------------------------
    variance_explained = eigvals_sorted ./ sum(eigvals_sorted)
    cum_explained = cumsum(variance_explained)

    k = findfirst(cum_explained .≥ percent_explained)
    Z =  PC_scores_sorted[:, 1:k]

    for i in 1:k
        println("PC$i: ", round(variance_explained[i]*100, digits=2), "%")
    end

    # ----------------------------
    # 5. Store PCA results in a DataFrame
    # ----------------------------
    pc_cols = Symbol.("PC", 1:size(Z,2))
    df_pca = DataFrame(Z, pc_cols)

    # Add row index mapping to original DataFrame
    df_pca[!,:row_idx] = nonmissing_idx

    return Dict("pca_df"=>df_pca, "eigen_values"=>eigvals_sorted, "eigen_vectors"=>eigvecs_sorted)

end




