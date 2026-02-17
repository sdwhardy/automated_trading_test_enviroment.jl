# Global Liquidity Factor Construction

A comprehensive Python framework for building a **global liquidity factor** usable across multiple risk assets (equities, crypto, commodities). The factor aggregates monetary policy signals from major economies using central bank balance sheets, monetary aggregates, credit growth, and cross-border USD funding proxies.

## 🎯 Overview

**Enhanced Version 2.0** with deep China & Japan coverage recognizing their systemic importance to global liquidity transmission.

This framework constructs a unified liquidity measure by:

1. **Data Collection**: Fetches **65+ indicators** from FRED, ECB SDMX, BIS, IMF, and OECD APIs
2. **Geographic Coverage**: US, Eurozone, **China (17 indicators)**, **Japan (19 indicators)**, plus global aggregates
3. **Standardization**: Converts all series to common frequency (weekly/monthly)
4. **Normalization**: Z-score normalization with rolling windows
5. **Factor Construction**: PCA, equal-weight, or custom-weighted aggregation

### Why Enhanced China & Japan Coverage?

**China:**
- World's 2nd largest economy (~18% global GDP)
- Total Social Financing (TSF) drives commodity demand
- Property market = 25% of GDP, massive liquidity channel
- Capital flows impact EM currencies and crypto

**Japan:**
- $5+ trillion BoJ balance sheet (130% of GDP)
- Yield Curve Control (YCC) = unlimited liquidity
- Yen carry trades fund global risk assets (~$500B)
- Only central bank buying equity ETFs

## 📊 Data Sources & Indicators

### United States (via FRED)
- **M2 Money Stock** - Broad money supply
- **Federal Reserve Assets** - Fed balance sheet total
- **Securities Held Outright** - QE program size
- **Bank Credit** - Commercial bank lending
- **Treasury General Account** - Government cash (reverse liquidity)
- **Overnight Reverse Repo** - Fed liquidity drain
- **Currency in Circulation**

### Eurozone (via FRED/ECB)
- **M2 Euro Area** - Eurozone broad money
- **ECB Assets** - ECB balance sheet
- **M2 YoY Growth**

### China (via FRED) - **17 Indicators**
- **Monetary**: M1, M2, M1/M2 ratio, growth rates
- **Central Bank**: PBoC assets, FX reserves
- **Credit**: Bank loans, domestic credit, total debt, TSF proxies
- **Rates**: Lending rate, 7-day repo
- **Activity**: Industrial production, retail sales, FDI
- **Derived**: Credit growth, M2 momentum

### Japan (via FRED) - **19 Indicators**
- **Monetary**: M2, M3, monetary base, growth rates
- **Central Bank**: BoJ assets, JGBs held, ETF holdings
- **Credit**: Bank lending, domestic credit, credit growth
- **YCC**: 10Y JGB yield, overnight call rate
- **Real**: Real M2, real M2 growth (inflation-adjusted)
- **External**: FX reserves, current account
- **Activity**: Industrial production, retail sales, CPI

### Global Indicators (via FRED/BIS)
- **USD Funding**: SOFR, LIBOR, TED spread, Fed swap lines, eurodollars
- **Credit Spreads**: High yield, investment grade, emerging markets (inverted)
- **Risk**: VIX (inverted), Dollar Index
- **Cross-border**: Offshore USD, global liquidity proxies

### OECD Backup (via FRED)
- **G7 Aggregates**: M1, credit
- **Other Economies**: UK M4, Canada M2, Australia M3
- **Trade**: World trade volume

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Get a FRED API Key

1. Visit https://fred.stlouisfed.org/docs/api/api_key.html
2. Create a free account
3. Generate your API key
4. Replace `YOUR_FRED_API_KEY_HERE` in the script

### 3. Run the Script

```python
from global_liquidity_factor import GlobalLiquidityFactor, Config

# Set your API key
Config.FRED_API_KEY = "your_actual_api_key_here"

# Initialize
glf = GlobalLiquidityFactor(Config.FRED_API_KEY)

# Fetch data and construct factor
data = glf.fetch_all_data()
factor = glf.construct_factor(method='pca')

# Save outputs
glf.save_data()
glf.save_factor()
glf.plot_factor()
glf.correlation_analysis()
```

## 📈 Methodology

### 1. Data Fetching
Each regional fetcher pulls data from official sources:
- `FREDFetcher` - US Federal Reserve Economic Data
- `ECBFetcher` - European Central Bank SDMX API
- `BISFetcher` - Bank for International Settlements
- `IMFFetcher` - International Monetary Fund

### 2. Frequency Standardization
All series resampled to common frequency (default: monthly)
```python
Config.TARGET_FREQ = "M"  # Monthly
# or
Config.TARGET_FREQ = "W"  # Weekly
```

### 3. Z-Score Normalization
Rolling window z-score to make indicators comparable:
```
z = (x - μ_rolling) / σ_rolling
```
Default window: 36 months (3 years)

### 4. Factor Construction

#### **Option A: PCA (Recommended)**
Extracts first principal component from all normalized indicators:
```python
factor = glf.construct_factor(method='pca', n_components=1)
```
- Automatically weights indicators by their contribution to variance
- Captures common liquidity signal
- Reports explained variance and component loadings

#### **Option B: Equal Weight**
Simple average of all normalized indicators:
```python
factor = glf.construct_factor(method='equal')
```

#### **Option C: Custom Weights**
User-defined weights for each indicator:
```python
weights = {
    'US_M2': 0.3,
    'US_FED_ASSETS': 0.2,
    'EZ_M2': 0.2,
    # ... etc
}
factor = glf.construct_factor(method='custom', weights=weights)
```

## 📁 Output Files

Running the script generates four files:

1. **`global_liquidity_data.csv`**
   - All indicators (z-score normalized)
   - Date index with all series as columns
   - Use for further analysis or backtesting

2. **`global_liquidity_factor.csv`**
   - Final composite liquidity factor
   - Single time series
   - Ready for use in trading models

3. **`global_liquidity_factor_chart.png`**
   - Visualization of factor over time
   - Shows expansionary (green) vs contractionary (red) periods
   - Includes 12-month moving average

4. **`liquidity_correlation_matrix.png`**
   - Heatmap of indicator correlations
   - Helps identify redundant or complementary indicators

## 🔧 Advanced Configuration

### Change Date Range
```python
Config.START_DATE = "2015-01-01"
Config.END_DATE = "2024-12-31"
```

### Change Normalization Window
```python
data_normalized = glf.normalize_zscore(data, window=24)  # 2-year window
```

### Add Custom Indicators
```python
# Add to fetch_us_liquidity() method
us_series['US_CUSTOM'] = 'YOUR_FRED_CODE'
```

### Export for Backtesting
```python
# Merge with asset returns
factor_df = pd.read_csv('results/global_liquidity_factor.csv', index_col=0, parse_dates=True)
returns_df = pd.read_csv('results/your_asset_returns.csv', index_col=0, parse_dates=True)
combined = pd.concat([factor_df, returns_df], axis=1).dropna()
```

## 📊 Interpretation Guide

### Factor Value Interpretation
- **Positive z-score**: Expansionary liquidity conditions
  - Central banks expanding balance sheets
  - Money supply growing faster than trend
  - Low credit spreads
  - Favorable for risk assets

- **Negative z-score**: Contractionary liquidity conditions
  - Central banks tightening or unwinding QE
  - Slowing money supply growth
  - Rising credit spreads
  - Risk-off environment

### Typical Use Cases

#### 1. **Asset Allocation**
```python
# Overweight risk assets when factor > 0.5
# Underweight when factor < -0.5
```

#### 2. **Risk-On/Risk-Off Signal**
```python
if factor.iloc[-1] > factor.rolling(12).mean().iloc[-1]:
    signal = "Risk-On"
else:
    signal = "Risk-Off"
```

#### 3. **Regime Identification**
```python
# High liquidity regime: factor > 1
# Normal regime: -1 < factor < 1  
# Tight liquidity regime: factor < -1
```

## 🌐 API Endpoints Reference

### FRED API
- **Base URL**: `https://api.stlouisfed.org/fred/series/observations`
- **Docs**: https://fred.stlouisfed.org/docs/api/fred/
- **Rate Limit**: No strict limit for registered users

### ECB SDMX API
- **Base URL**: `https://data-api.ecb.europa.eu/service/data`
- **Docs**: https://data.ecb.europa.eu/help/api/overview
- **Format**: SDMX-JSON

### BIS API
- **Base URL**: `https://data.bis.org/api/v1`
- **Docs**: https://www.bis.org/statistics/api_documentation.htm
- **Rate Limit**: Not publicly documented

### IMF API
- **Base URL**: `http://dataservices.imf.org/REST/SDMX_JSON.svc`
- **Docs**: https://datahelp.imf.org/knowledgebase/articles/667681
- **Format**: SDMX-JSON

## 🛠️ Troubleshooting

### Issue: API Request Fails
```
Error fetching [series]: HTTPError 403
```
**Solution**: Check your FRED API key is valid and properly set

### Issue: Missing Data
```
No valid data after cleaning
```
**Solution**: 
- Check date range isn't too recent (some series lag)
- Try increasing `Config.START_DATE` to 2010 or earlier
- Some series may be discontinued

### Issue: Import Error for sklearn
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: 
```bash
pip install scikit-learn
```

## 📚 Further Reading

### Academic Research
- **Hoerova, M., et al. (2018)**: "Money Markets and Bank Lending"
- **Adrian, T., Shin, H.S. (2010)**: "Liquidity and Leverage"
- **Brana, S., Prat, S. (2016)**: "The Effects of Global Excess Liquidity"

### Data Documentation
- [FRED Data Dictionary](https://fred.stlouisfed.org/)
- [ECB Statistical Data Warehouse](https://sdw.ecb.europa.eu/)
- [BIS Statistics Explorer](https://stats.bis.org/statx/toc/LBS.html)

### Quant Finance Resources
- [QuantLib Python](https://www.quantlib.org/)
- [Zipline Backtesting](https://github.com/quantopian/zipline)
- [Backtrader](https://www.backtrader.com/)

## 🤝 Contributing

Potential enhancements:
- [ ] Add UK/Switzerland central bank data
- [ ] Include repo/secured funding metrics
- [ ] Add cryptocurrency on-chain liquidity metrics
- [ ] Implement Kalman filter for real-time updates
- [ ] Add machine learning for dynamic weighting

## ⚖️ License

This project is released under the MIT License. See LICENSE file for details.

## ⚠️ Disclaimer

This tool is for **research and educational purposes only**. It is not investment advice. Past liquidity conditions do not guarantee future asset returns. Always consult with a qualified financial advisor before making investment decisions.

---

**Author**: Claude (Anthropic)  
**Version**: 1.0  
**Last Updated**: February 2026
