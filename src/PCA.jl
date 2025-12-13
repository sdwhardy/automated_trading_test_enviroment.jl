using DataFrames
using Statistics
using LinearAlgebra

# ----------------------------
# 1. Define your feature columns
# ----------------------------
feature_cols = [
    :lnReturn1day,
    :lnReturn5day,
    :lnReturn21day,
    :roc21day,
    :roc63day,
    :realVol5day,
    :realVol10day,
    :realVol21day,
    :gkVol21day,
    :volOfVol21day,
    :maDiff10_50day,
    :maDiff20_100day,
    :volumeZscore21day,
    :amihud21day
]

# ----------------------------
# 2. Extract DataFrame
# ----------------------------
df = OHLCVT["XBTUSD"]["1440"]["df"]

# ----------------------------
# 3. Identify rows with no missing values in features
# ----------------------------
X_all = Matrix(df[:, feature_cols])
row_mask = all(.!ismissing.(X_all), dims=2)   # true if row has no missing
row_mask_vec = vec(row_mask)
nonmissing_idx = findall(row_mask_vec)

# Subset DataFrame and matrix
df_nonmissing = df[nonmissing_idx, :]
X = Matrix(df_nonmissing[:, feature_cols])

# ----------------------------
# 4. Center the data (features are already Z-scored, but centering is safe)
# ----------------------------
X_centered = X .- mean(X, dims=1)

# ----------------------------
# 5. Compute PCA via covariance and eigen decomposition
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
# 6. Store PCA results in a DataFrame
# ----------------------------
pc_cols = Symbol.("PC", 1:size(PC_scores_sorted,2))
df_pca = DataFrame(PC_scores_sorted, pc_cols)

# Add row index mapping to original DataFrame
df_pca[!,:row_idx] = nonmissing_idx

# ----------------------------
# 7. Optional: Variance explained by each PC
# ----------------------------
variance_explained = eigvals_sorted ./ sum(eigvals_sorted)
cum_explained = cumsum(variance_explained)

k = findfirst(cum_explained .≥ 0.95)
Z = PC_scores[:, 1:k]

for i in 1:k
    println("PC$i: ", round(variance_explained[i]*100, digits=2), "%")
end


###########################################

tol = 1e-8

println("PCA validation checks:")

# ----------------------------
# 1. Eigenvalues non-negative
# ----------------------------
check1 = minimum(eigvals_sorted) ≥ -tol
println("1. Eigenvalues non-negative: ", check1)

# ----------------------------
# 2. Eigenvectors orthonormal
# ----------------------------
I_approx = eigvecs_sorted' * eigvecs_sorted
check2 = norm(I_approx - I, Inf) < tol
println("2. Eigenvectors orthonormal: ", check2)

# ----------------------------
# 3. Variance of PC scores = eigenvalues
# ----------------------------
pc_vars = vec(var(PC_scores_sorted; dims=1))
check3 = norm(pc_vars - eigvals_sorted, Inf) < tol
println("3. PC variances match eigenvalues: ", check3)

# ----------------------------
# 4. Reconstruction error
# ----------------------------
X_reconstructed = PC_scores_sorted * eigvecs_sorted'
recon_error = norm(X_reconstructed - X_centered, Inf)
check4 = recon_error < tol
println("4. Reconstruction accurate: ", check4)

println("\nMax reconstruction error = ", recon_error)

