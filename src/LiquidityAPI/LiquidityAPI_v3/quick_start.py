"""
Quick Start — Global Liquidity Factor (Block-PCA)
==================================================
Three steps:
    1. Set your FRED API key below
    2. Run:  python quick_start.py
    3. Outputs land in results/

API key (free): https://fred.stlouisfed.org/docs/api/api_key.html
"""

from global_liquidity_factor import GlobalLiquidityFactor, Config

# ── SET YOUR KEY HERE ──────────────────────────────────────────────────────────
#Config.FRED_API_KEY = "YOUR_FRED_API_KEY_HERE"
# ──────────────────────────────────────────────────────────────────────────────


def quick_start():

    api_key = Config.FRED_API_KEY

    if api_key == "YOUR_FRED_API_KEY_HERE":
        print("\n⚠️  FRED API key not set!")
        print("   Edit quick_start.py and replace YOUR_FRED_API_KEY_HERE\n")
        return

    print("=" * 60)
    print("  GLOBAL LIQUIDITY FACTOR  —  Block-PCA v3")
    print("=" * 60)

    glf = GlobalLiquidityFactor(api_key)

    # 1. Fetch & normalise all raw indicators (~2 min)
    print("\nStep 1/3  Fetching indicators from FRED ...")
    data = glf.fetch_all_data()
    if data.empty:
        print("No data fetched. Check API key and internet connection.")
        return
    print(f"  {len(data.columns)} indicators loaded  "
          f"({data.index.min().strftime('%Y-%m')} -> {data.index.max().strftime('%Y-%m')})")

    glf.save_data()   # -> results/global_liquidity_data.csv

    # 2. Block-PCA -> 6 scores across 5 thematic blocks
    print("\nStep 2/3  Running block-PCA ...")
    scores = glf.construct_block_factors()
    # scores columns:
    #   Monetary_Aggregates_PC1, Monetary_Aggregates_PC2
    #   CB_Balance_Sheets_PC1
    #   Credit_Spreads_PC1
    #   FX_Commodities_PC1
    #   Macro_Flows_PC1

    # 3. Charts
    print("\nStep 3/3  Generating charts ...")
    glf.plot_block_factors()        # -> results/block_pca_scores.png
    glf.plot_block_correlation()    # -> results/block_pca_correlation.png

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print("\nFiles written to results/")
    print("  global_liquidity_data.csv    -- all normalised indicators")
    print("  block_pca_scores.csv         -- 6 block scores (time series)")
    print("  block_pca_loadings.csv       -- PCA loadings + explained variance")
    print("  block_pca_scores.png         -- 5-panel factor chart")
    print("  block_pca_correlation.png    -- inter-block correlation heatmap")

    print(f"\nLatest readings  ({scores.index[-1].strftime('%Y-%m-%d')}):\n")
    for col, val in scores.iloc[-1].items():
        arrow  = "^" if val > 0 else "v"
        regime = ("Expansionary"  if val >  0.5
                  else "Contractionary" if val < -0.5
                  else "Neutral")
        print(f"  {arrow} {col:<35s}  {val:+.3f}  [{regime}]")


if __name__ == "__main__":
    quick_start()
