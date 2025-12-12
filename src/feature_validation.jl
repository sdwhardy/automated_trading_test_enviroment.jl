using Statistics
using DataFrames
using Printf


"""
validate_features(df; std_tol=1e-3, mean_tol=1e-3, plot=false)

Checks:
  - mean ≈ 0
  - std ≈ 1
  - missing values
  - histogram (optional)
  - alignment: no missing in middle of series

Returns:
  A DataFrame summary of pass/fail per feature.
"""
function validate_features(df::DataFrame; std_tol=1e-3, mean_tol=1e-3)

    results = DataFrame(
        feature = String[],
        mean     = Float64[],
        std      = Float64[],
        missing  = Int[],
        mean_ok  = Bool[],
        std_ok   = Bool[],
        aligned  = Bool[]
    )

    for col in names(df)
        x = df[!, col]
        clean = skipmissing(x) |> collect

        μ = mean(clean)
        σ = std(clean)
        missing_count = count(ismissing, x)

        mean_ok = abs(μ) ≤ mean_tol
        std_ok  = abs(σ - 1) ≤ std_tol

        # Alignment test:
        # Missing values allowed only at the start, not in middle.
        first_valid = findfirst(!ismissing, x)
        last_missing_after_valid = any(ismissing.(x[first_valid:end]))

        aligned_ok = !last_missing_after_valid

        push!(results, (String(col), μ, σ, missing_count, mean_ok, std_ok, aligned_ok))

    end

    return results
end
