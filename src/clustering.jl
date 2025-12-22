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
            gmm = GMM(k, X; kind=kind)
            em!(gmm, X; nIter=50)
            push!(bics, gmm_bic(gmm, X))
            push!(models, gmm)
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

