"""
    PCA(df::DataFrame, feature_cols::Vector{Symbol}) -> Dict{String, Any}

Compute Principal Component Analysis (PCA) on a set of preprocessed features.

This implementation assumes features are already **Z-score normalized**.
Centering is still applied explicitly to ensure numerical robustness.

### Steps
1. **Missing-value filtering**
   Rows containing any missing values in `feature_cols` are removed.
   The indices of retained rows are preserved.

2. **Centering**
   Each feature column is centered by subtracting its sample mean.

3. **Covariance-based PCA**
   - Compute the sample covariance matrix of the centered data.
   - Perform eigenvalue decomposition of the covariance matrix.
   - Sort eigenvalues and eigenvectors in descending order.

4. **Projection**
   Project the centered data onto the ordered eigenvectors to obtain
   principal component scores.

### Returns
A dictionary with the following keys:
- `"pca_df"`          : Matrix of principal component scores
- `"eigen_values"`   : Sorted eigenvalues (explained variances)
- `"eigen_vectors"`  : Corresponding eigenvectors (principal directions)
- `"nonmissing_idx"` : Row indices of observations used in PCA
- `"X_centered"`     : Centered feature matrix used for decomposition

### Notes
- Eigenvalues correspond to the variance of each principal component.
- Eigenvectors are orthonormal by construction.
- Reconstruction is possible via `pca_df * eigen_vectors'`.

### References
- Jolliffe, I. T. *Principal Component Analysis*, Springer.
  https://link.springer.com/book/10.1007/978-1-4757-1904-8
"""
function PCA(df::DataFrame, feature_cols::Vector{Symbol})


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


    return Dict("pca_df"=>PC_scores_sorted, "eigen_values"=>eigvals_sorted, "eigen_vectors"=>eigvecs_sorted, "nonmissing_idx"=>nonmissing_idx, "X_centered"=>X_centered)

end

function percent_explained_PCA(PC_scores_sorted,eigvals_sorted, nonmissing_idx; percent_explained::Float64=0.95)
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

    return df_pca
end
