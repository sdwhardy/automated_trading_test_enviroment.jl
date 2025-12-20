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

function num_params_full(k, d)
    # weights (k-1) + means (k*d) + covariances (k*d*(d+1)/2)
    return (k - 1) + k*d + k*d*(d + 1) ÷ 2
end


function mean_entropy(post)
    return mean(-sum(p .* log.(p .+ eps())) for p in eachrow(post))
end

