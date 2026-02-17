"""
Advanced Usage Examples for Global Liquidity Factor
===================================================
This module demonstrates advanced applications including:
- Backtesting against asset returns
- Multi-asset correlation analysis
- Regime-based strategy
- Rolling factor construction
- Out-of-sample validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from global_liquidity_factor import GlobalLiquidityFactor, Config


# ============================================================================
# EXAMPLE 1: Backtest Factor Against Asset Returns
# ============================================================================

def backtest_factor_strategy():
    """
    Backtest a simple strategy:
    - Long risk assets when liquidity factor > 0
    - Short/cash when liquidity factor < 0
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Backtesting Liquidity Factor Strategy")
    print("="*60)
    
    # Load the factor
    factor = pd.read_csv('results/global_liquidity_factor.csv', 
                         index_col=0, parse_dates=True)
    factor.columns = ['Liquidity_Factor']
    
    # For demonstration, create synthetic asset returns
    # In practice, you'd load real returns (S&P 500, Bitcoin, Gold, etc.)
    np.random.seed(42)
    dates = factor.index
    
    # Synthetic returns correlated with liquidity
    base_return = 0.0005  # 0.05% daily
    noise = np.random.randn(len(dates)) * 0.01
    liquidity_effect = factor['Liquidity_Factor'].values * 0.003
    
    asset_returns = pd.DataFrame({
        'SPX': base_return + liquidity_effect + noise,
        'BTC': base_return + liquidity_effect * 1.5 + noise * 2,
        'GOLD': base_return + liquidity_effect * 0.5 + noise * 0.5,
    }, index=dates)
    
    # Strategy signals
    signals = pd.DataFrame(index=dates)
    signals['Long_Signal'] = (factor['Liquidity_Factor'] > 0).astype(int)
    signals['Short_Signal'] = (factor['Liquidity_Factor'] < 0).astype(int)
    
    # Calculate strategy returns
    strategy_returns = pd.DataFrame(index=dates)
    strategy_returns['SPX_Strategy'] = asset_returns['SPX'] * signals['Long_Signal']
    strategy_returns['BTC_Strategy'] = asset_returns['BTC'] * signals['Long_Signal']
    strategy_returns['GOLD_Strategy'] = asset_returns['GOLD'] * signals['Long_Signal']
    strategy_returns['Buy_Hold_SPX'] = asset_returns['SPX']
    
    # Calculate cumulative returns
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # Plot results
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Cumulative returns
    axes[0].plot(cumulative_returns.index, cumulative_returns['SPX_Strategy'], 
                label='SPX Strategy (Liquidity-Timed)', linewidth=2)
    axes[0].plot(cumulative_returns.index, cumulative_returns['Buy_Hold_SPX'], 
                label='SPX Buy & Hold', linewidth=2, linestyle='--', alpha=0.7)
    axes[0].set_title('Liquidity-Timed Strategy vs Buy & Hold', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Return', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Drawdown
    drawdown = (cumulative_returns / cumulative_returns.cummax() - 1) * 100
    axes[1].fill_between(drawdown.index, 0, drawdown['SPX_Strategy'], 
                        alpha=0.3, color='red', label='SPX Strategy')
    axes[1].fill_between(drawdown.index, 0, drawdown['Buy_Hold_SPX'], 
                        alpha=0.3, color='blue', label='SPX Buy & Hold')
    axes[1].set_title('Strategy Drawdowns', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown (%)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/backtest_results.png', dpi=300, bbox_inches='tight')
    print("✓ Backtest chart saved to backtest_results.png")
    
    # Performance metrics
    total_return_strategy = (cumulative_returns['SPX_Strategy'].iloc[-1] - 1) * 100
    total_return_bh = (cumulative_returns['Buy_Hold_SPX'].iloc[-1] - 1) * 100
    sharpe_strategy = strategy_returns['SPX_Strategy'].mean() / strategy_returns['SPX_Strategy'].std() * np.sqrt(252)
    sharpe_bh = strategy_returns['Buy_Hold_SPX'].mean() / strategy_returns['Buy_Hold_SPX'].std() * np.sqrt(252)
    
    print(f"\nPerformance Metrics:")
    print(f"Strategy Total Return: {total_return_strategy:.2f}%")
    print(f"Buy & Hold Total Return: {total_return_bh:.2f}%")
    print(f"Strategy Sharpe Ratio: {sharpe_strategy:.2f}")
    print(f"Buy & Hold Sharpe Ratio: {sharpe_bh:.2f}")


# ============================================================================
# EXAMPLE 2: Multi-Asset Correlation Analysis
# ============================================================================

def analyze_factor_asset_correlations():
    """
    Analyze rolling correlations between liquidity factor and various assets
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Multi-Asset Correlation Analysis")
    print("="*60)
    
    # Load factor
    factor = pd.read_csv('results/global_liquidity_factor.csv', 
                         index_col=0, parse_dates=True)
    
    # Create synthetic asset data (in practice, load real data)
    np.random.seed(42)
    dates = factor.index
    
    assets = pd.DataFrame({
        'Equities': np.random.randn(len(dates)).cumsum() + factor.values.flatten() * 10,
        'Credit': np.random.randn(len(dates)).cumsum() + factor.values.flatten() * 8,
        'Commodities': np.random.randn(len(dates)).cumsum() + factor.values.flatten() * 6,
        'EM_FX': np.random.randn(len(dates)).cumsum() + factor.values.flatten() * 7,
        'Crypto': np.random.randn(len(dates)).cumsum() + factor.values.flatten() * 12,
    }, index=dates)
    
    # Calculate rolling correlations
    window = 36  # 3 years
    rolling_corrs = pd.DataFrame(index=dates)
    
    for col in assets.columns:
        rolling_corrs[col] = factor.rolling(window).corr(assets[col])
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    for col in rolling_corrs.columns:
        ax.plot(rolling_corrs.index, rolling_corrs[col], label=col, linewidth=2)
    
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_title(f'Rolling {window}-Month Correlations: Liquidity Factor vs Assets', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Correlation Coefficient', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/asset_correlations.png', dpi=300, bbox_inches='tight')
    print("✓ Correlation chart saved to asset_correlations.png")
    
    # Print current correlations
    print(f"\nCurrent Correlations (latest {window} months):")
    for col in assets.columns:
        corr = rolling_corrs[col].iloc[-1]
        print(f"{col:15s}: {corr:6.2f}")


# ============================================================================
# EXAMPLE 3: Liquidity Regime Identification
# ============================================================================

def identify_liquidity_regimes():
    """
    Classify periods into liquidity regimes and analyze asset behavior
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Liquidity Regime Identification")
    print("="*60)
    
    # Load factor
    factor = pd.read_csv('results/global_liquidity_factor.csv', 
                         index_col=0, parse_dates=True)
    factor.columns = ['Factor']
    
    # Define regimes based on factor value and trend
    factor['Regime'] = 'Normal'
    factor.loc[factor['Factor'] > 1, 'Regime'] = 'High_Liquidity'
    factor.loc[factor['Factor'] < -1, 'Regime'] = 'Tight_Liquidity'
    
    # Add momentum signal
    factor['Momentum'] = factor['Factor'] - factor['Factor'].shift(6)  # 6-month change
    factor.loc[(factor['Regime'] == 'Normal') & (factor['Momentum'] > 0.5), 'Regime'] = 'Expanding'
    factor.loc[(factor['Regime'] == 'Normal') & (factor['Momentum'] < -0.5), 'Regime'] = 'Contracting'
    
    # Plot regimes
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Color map for regimes
    regime_colors = {
        'High_Liquidity': 'darkgreen',
        'Expanding': 'lightgreen',
        'Normal': 'gray',
        'Contracting': 'orange',
        'Tight_Liquidity': 'darkred'
    }
    
    for regime, color in regime_colors.items():
        mask = factor['Regime'] == regime
        if mask.any():
            ax.scatter(factor.index[mask], factor['Factor'][mask], 
                      label=regime.replace('_', ' '), color=color, alpha=0.6, s=20)
    
    ax.plot(factor.index, factor['Factor'], color='black', alpha=0.2, linewidth=1)
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axhline(y=1, color='green', linestyle=':', alpha=0.3)
    ax.axhline(y=-1, color='red', linestyle=':', alpha=0.3)
    ax.set_title('Liquidity Regime Classification', fontsize=14, fontweight='bold')
    ax.set_ylabel('Liquidity Factor (Z-Score)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/liquidity_regimes.png', dpi=300, bbox_inches='tight')
    print("✓ Regime chart saved to liquidity_regimes.png")
    
    # Regime statistics
    print("\nRegime Distribution:")
    regime_counts = factor['Regime'].value_counts()
    for regime, count in regime_counts.items():
        pct = (count / len(factor)) * 100
        print(f"{regime:20s}: {count:4d} periods ({pct:5.1f}%)")


# ============================================================================
# EXAMPLE 4: Out-of-Sample Validation
# ============================================================================

def rolling_factor_validation():
    """
    Construct factor with expanding window to simulate real-time usage
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Rolling Out-of-Sample Validation")
    print("="*60)
    
    # This would require re-fetching data with different windows
    # For demonstration, we'll show the concept
    
    print("\nConcept: Rolling Window Factor Construction")
    print("-" * 60)
    print("1. Split data into training (historical) and testing (recent)")
    print("2. Construct factor using only training data")
    print("3. Apply same normalization parameters to test data")
    print("4. Evaluate predictive power out-of-sample")
    print("\nImplementation requires re-running factor construction")
    print("with different date ranges. See main script for details.")


# ============================================================================
# EXAMPLE 5: Custom Indicator Weighting
# ============================================================================

def custom_weighted_factor():
    """
    Build factor with custom weights emphasizing specific indicators
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Custom Weighted Factor Construction")
    print("="*60)
    
    # Initialize factor builder
    glf = GlobalLiquidityFactor(Config.FRED_API_KEY)
    
    # Fetch data
    data = glf.fetch_all_data()
    
    if not data.empty:
        # Define custom weights (emphasize central bank balance sheets)
        custom_weights = {
            'US_FED_ASSETS': 0.25,
            'US_FED_SECURITIES': 0.15,
            'EZ_ECB_ASSETS': 0.20,
            'CN_PBOC_ASSETS': 0.15,
            'JP_BOJ_ASSETS': 0.15,
            'US_M2': 0.05,
            'EZ_M2': 0.05,
        }
        
        print("\nCustom Weights:")
        for indicator, weight in custom_weights.items():
            print(f"{indicator:25s}: {weight:.2%}")
        
        # Construct factor
        factor_custom = glf.construct_factor(method='custom', weights=custom_weights)
        
        # Compare with PCA factor
        factor_pca = glf.construct_factor(method='pca')
        
        # Plot comparison
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        axes[0].plot(factor_pca.index, factor_pca.values, label='PCA Factor', linewidth=2)
        axes[0].plot(factor_custom.index, factor_custom.values, label='Custom Weighted', linewidth=2, alpha=0.7)
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0].set_title('Factor Comparison: PCA vs Custom Weights', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Factor Value', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot difference
        difference = factor_pca - factor_custom
        axes[1].plot(difference.index, difference.values, color='purple', linewidth=2)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1].fill_between(difference.index, 0, difference.values, alpha=0.3, color='purple')
        axes[1].set_title('Difference: PCA - Custom', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Difference', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/mnt/user-data/outputs/factor_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Comparison chart saved to factor_comparison.png")
        
        # Calculate correlation between factors
        corr = factor_pca.corr(factor_custom)
        print(f"\nCorrelation between PCA and Custom: {corr:.3f}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all advanced examples"""
    
    print("="*60)
    print("ADVANCED USAGE EXAMPLES - GLOBAL LIQUIDITY FACTOR")
    print("="*60)
    
    # Check if factor file exists
    try:
        pd.read_csv('results/global_liquidity_factor.csv', nrows=1)
    except FileNotFoundError:
        print("\n⚠️  Factor file not found!")
        print("Please run global_liquidity_factor.py first to generate the factor.\n")
        return
    
    # Run examples
    try:
        backtest_factor_strategy()
    except Exception as e:
        print(f"\nError in Example 1: {str(e)}")
    
    try:
        analyze_factor_asset_correlations()
    except Exception as e:
        print(f"\nError in Example 2: {str(e)}")
    
    try:
        identify_liquidity_regimes()
    except Exception as e:
        print(f"\nError in Example 3: {str(e)}")
    
    try:
        rolling_factor_validation()
    except Exception as e:
        print(f"\nError in Example 4: {str(e)}")
    
    try:
        custom_weighted_factor()
    except Exception as e:
        print(f"\nError in Example 5: {str(e)}")
    
    print("\n" + "="*60)
    print("✓ ALL EXAMPLES COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("1. backtest_results.png")
    print("2. asset_correlations.png")
    print("3. liquidity_regimes.png")
    print("4. factor_comparison.png")


if __name__ == "__main__":
    main()
