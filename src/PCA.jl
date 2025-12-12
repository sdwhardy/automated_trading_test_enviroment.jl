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

# Principal component scores
PC_scores = X_centered * eigvecs

# ----------------------------
# 6. Store PCA results in a DataFrame
# ----------------------------
pc_cols = Symbol.("PC", 1:size(PC_scores,2))
df_pca = DataFrame(PC_scores, pc_cols)

# Add row index mapping to original DataFrame
df_pca[!,:row_idx] = nonmissing_idx

# ----------------------------
# 7. Optional: Variance explained by each PC
# ----------------------------
variance_explained = eigvals ./ sum(eigvals)
