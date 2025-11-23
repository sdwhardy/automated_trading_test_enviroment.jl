using Documenter

# Include your package here (the package you are documenting)
using automated_trading_test_environment

# Set up documentation
build_dir = "../docs/build"

# Check if the build directory exists and try to delete it
if isdir(build_dir)
    try
        rm(build_dir; force=true, recursive=true)
    catch e
        @warn "Failed to delete build directory: $e"
    end
end

# Set up documentation
makedocs(
    # Specify the format (HTML or PDF)
    sitename = "automated_trading_test_environment Documentation",
    modules = [automated_trading_test_environment],
    format = Documenter.HTML(),
    # Directory to output the docs
    build = "../docs/build"
)