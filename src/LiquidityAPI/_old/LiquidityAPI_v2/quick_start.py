"""
Quick Start Guide - Global Liquidity Factor
==========================================
Run this script to get started quickly with the liquidity factor
"""

from global_liquidity_factor import GlobalLiquidityFactor, Config

def quick_start():
    """Minimal example to get the liquidity factor"""
    
    print("="*60)
    print("QUICK START - GLOBAL LIQUIDITY FACTOR")
    print("="*60)
    
    # Step 1: Set your FRED API key
    print("\nStep 1: Setting up API key...")
    api_key = Config.FRED_API_KEY
    
    if api_key == "YOUR_FRED_API_KEY_HERE":
        print("\n⚠️  You need to set your FRED API key first!")
        print("\n1. Get a free key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Open global_liquidity_factor.py")
        print("3. Replace 'YOUR_FRED_API_KEY_HERE' with your actual key")
        print("4. Run this script again\n")
        return
    
    print("✓ API key configured")
    
    # Step 2: Initialize
    print("\nStep 2: Initializing factor builder...")
    glf = GlobalLiquidityFactor(api_key)
    print("✓ Builder initialized")
    
    # Step 3: Fetch data
    print("\nStep 3: Fetching liquidity data from FRED, ECB, BIS, IMF...")
    print("(This may take 1-2 minutes depending on your connection)")
    data = glf.fetch_all_data()
    
    if data.empty:
        print("❌ Failed to fetch data. Check your API key and internet connection.")
        return
    
    print(f"✓ Fetched {len(data.columns)} indicators")
    print(f"✓ Date range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
    
    # Step 4: Construct factor
    print("\nStep 4: Constructing liquidity factor using PCA...")
    factor = glf.construct_factor(method='pca')
    print(f"✓ Factor constructed ({len(factor)} observations)")
    
    # Step 5: Save outputs
    print("\nStep 5: Saving outputs...")
    glf.save_data('results/global_liquidity_data.csv')
    glf.save_factor('results/global_liquidity_factor.csv')
    glf.plot_factor()
    glf.correlation_analysis()
    print("✓ All outputs saved")
    
    # Step 6: Summary
    print("\n" + "="*60)
    print("🎉 SUCCESS! Your liquidity factor is ready!")
    print("="*60)
    
    print("\n📊 Factor Summary:")
    print(f"Current Value: {factor.iloc[-1]:.3f}")
    print(f"6M Average:    {factor.iloc[-6:].mean():.3f}")
    print(f"12M Average:   {factor.iloc[-12:].mean():.3f}")
    
    if factor.iloc[-1] > 0:
        print("\n💡 Interpretation: EXPANSIONARY liquidity conditions")
        print("   → Favorable environment for risk assets")
    else:
        print("\n💡 Interpretation: CONTRACTIONARY liquidity conditions")
        print("   → Challenging environment for risk assets")
    
    print("\n📁 Output Files:")
    print("   1. global_liquidity_data.csv - Raw indicators")
    print("   2. global_liquidity_factor.csv - The factor itself")
    print("   3. global_liquidity_factor_chart.png - Visualization")
    print("   4. liquidity_correlation_matrix.png - Correlation heatmap")
    
    print("\n🚀 Next Steps:")
    print("   - Review the generated charts")
    print("   - Run advanced_usage.py for backtesting examples")
    print("   - Integrate the factor into your trading models")
    print("   - Update regularly to track liquidity conditions")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    quick_start()
