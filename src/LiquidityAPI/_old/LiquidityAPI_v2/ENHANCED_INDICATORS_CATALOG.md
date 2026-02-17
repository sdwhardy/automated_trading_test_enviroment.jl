# Enhanced Global Liquidity Indicators Catalog

## Overview
This enhanced version includes **50+ indicators** with deep coverage of China and Japan, recognizing their systemic importance to global liquidity and regional capital transmission.

---

## 🇨🇳 CHINA INDICATORS (Comprehensive)

### Why China Matters
- **World's 2nd largest economy** (~18% of global GDP)
- **Largest manufacturing base** - drives commodity demand
- **Total Social Financing (TSF)** - massive credit creation outside banking system
- **Property market** - 25% of GDP, major liquidity channel
- **Capital controls** - domestic liquidity spills to offshore assets

### Monetary Aggregates
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| M2 Money Supply | `MYAGM2CNM189N` | Broad money including deposits | Monthly | ⭐⭐⭐⭐⭐ |
| M1 Money Supply | `MABMM301CNM189S` | Narrow money (cash + demand deposits) | Monthly | ⭐⭐⭐⭐ |
| M2 YoY Growth | Calculated | Annual growth rate | Monthly | ⭐⭐⭐⭐⭐ |
| M1 YoY Growth | Calculated | Annual growth rate | Monthly | ⭐⭐⭐⭐ |
| M1/M2 Ratio | Calculated | Liquidity velocity indicator | Monthly | ⭐⭐⭐⭐ |

**Interpretation:**
- **M2 > 10% YoY**: Expansionary (supports commodities, EM assets)
- **M1/M2 rising**: Money moving from savings to transactions (bullish)
- **M1/M2 falling**: Money parking in time deposits (bearish)

### Central Bank & Reserves
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| PBoC Total Assets | `DDDI06CNA156NWDB` | People's Bank balance sheet | Monthly | ⭐⭐⭐⭐⭐ |
| Foreign Reserves | `TRESEGCNM052N` | FX reserves (USD, EUR, JPY) | Monthly | ⭐⭐⭐⭐⭐ |

**Interpretation:**
- **Reserves declining**: Capital outflows, tighter liquidity
- **PBoC expanding**: Monetary easing, RRR cuts
- **Reserves > $3.2T**: Comfortable, can defend CNY

### Credit Aggregates
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| Bank Loans Outstanding | `CHNBSLARNADSMEI` | Total bank lending | Monthly | ⭐⭐⭐⭐⭐ |
| Domestic Credit | `QCNM770A` | Credit to private sector | Quarterly | ⭐⭐⭐⭐ |
| Total Debt | `QCNPAM770A` | Total private non-financial debt | Quarterly | ⭐⭐⭐⭐⭐ |
| Credit Growth YoY | Calculated | Annual loan growth | Monthly | ⭐⭐⭐⭐⭐ |

**Key Concept: Total Social Financing (TSF)**
- TSF = Bank loans + Shadow banking + Corporate bonds + Equity issuance
- **China-specific liquidity metric** (not captured in M2)
- When TSF > Bank loans growth: Shadow banking expanding

### Interest Rates & Policy
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| Lending Rate | `INTGSTCNM193N` | Average bank lending rate | Monthly | ⭐⭐⭐ |
| 7-Day Repo Rate | `IRCN7DV01STM` | Interbank funding cost | Daily | ⭐⭐⭐⭐ |

**PBoC Policy Tools:**
- **RRR (Reserve Requirement Ratio)**: Primary liquidity tool
- **MLF (Medium-term Lending Facility)**: Liquidity to banks
- **IOER (Interest on Excess Reserves)**: Floor rate

### Economic Activity (Liquidity Demand)
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| Industrial Production | `CHNTOTINDFODISMEI` | Manufacturing output | Monthly | ⭐⭐⭐ |
| Retail Sales | `SLRTTO01CNM661S` | Consumer spending | Monthly | ⭐⭐⭐ |
| FDI Flows | `BPFAIN01CNQ188S` | Foreign investment | Quarterly | ⭐⭐⭐ |

---

## 🇯🇵 JAPAN INDICATORS (Comprehensive)

### Why Japan Matters
- **World's 3rd largest economy**
- **$5+ trillion central bank balance sheet** - largest in G7 relative to GDP
- **Yield Curve Control (YCC)** - unlimited liquidity to cap rates
- **Carry trade hub** - zero rates fund global risk assets
- **ETF buying** - BoJ owns 7% of Japanese equity market

### Monetary Aggregates
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| M2 Money Supply | `MYAGM2JPM189N` | Broad money | Monthly | ⭐⭐⭐⭐⭐ |
| M3 Money Supply | `MABMM301JPM189S` | Broadest measure | Monthly | ⭐⭐⭐⭐ |
| Monetary Base | `JPNBASMON` | Cash + reserves | Monthly | ⭐⭐⭐⭐ |
| M2 YoY Growth | Calculated | Annual growth | Monthly | ⭐⭐⭐⭐⭐ |
| M3 YoY Growth | Calculated | Annual growth | Monthly | ⭐⭐⭐⭐ |

**Interpretation:**
- **M2 > 5% YoY**: Highly expansionary for Japan (normal 2-3%)
- **Monetary Base >> M2**: Quantitative easing in effect

### Bank of Japan Balance Sheet
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| BoJ Total Assets | `JPNASSETS` | Central bank balance sheet | Monthly | ⭐⭐⭐⭐⭐ |
| JGBs Held | `JPNJGBAQ` | Japanese govt bonds owned | Quarterly | ⭐⭐⭐⭐⭐ |
| ETF Holdings | `JPNETFAQ` | Equity ETFs owned (unique!) | Quarterly | ⭐⭐⭐⭐⭐ |
| BoJ Growth YoY | Calculated | Annual balance sheet growth | Monthly | ⭐⭐⭐⭐⭐ |

**Why BoJ is Unique:**
- **YCC**: BoJ buys unlimited JGBs to keep 10Y yield at target
- **ETF purchases**: Direct equity market support
- **Balance sheet = 130% of GDP** (vs 35% for Fed)

### Credit & Lending
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| Bank Lending | `JPNBSLARNADSMEI` | Total bank loans | Monthly | ⭐⭐⭐⭐ |
| Domestic Credit | `QJPM770A` | Credit to private sector | Quarterly | ⭐⭐⭐⭐ |
| Credit Growth YoY | Calculated | Annual loan growth | Monthly | ⭐⭐⭐⭐ |

### Interest Rates & YCC
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| 10-Year JGB Yield | `IRLTLT01JPM156N` | YCC target rate | Daily | ⭐⭐⭐⭐⭐ |
| Overnight Call Rate | `IRSTCI01JPM156N` | Policy rate | Daily | ⭐⭐⭐⭐ |

**YCC Framework:**
- **Target**: 10Y yield at ~0% (±0.5%)
- **Implication**: Unlimited liquidity to hit target
- **Exit risk**: If BoJ abandons YCC, massive liquidity withdrawal

### Currency & Cross-Border
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| FX Reserves | `TRESEGJPM052N` | Foreign currency reserves | Monthly | ⭐⭐⭐ |
| Current Account | `JPNBCA` | Trade + investment balance | Monthly | ⭐⭐⭐⭐ |

**Yen Carry Trade:**
- **Mechanics**: Borrow JPY at 0%, invest in higher-yielding assets
- **Unwind risk**: When JPY strengthens, positions forced to close
- **Global impact**: Estimated $500B+ in carry trades

### Inflation & Real Liquidity
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| CPI All Items | `JPNCPIALLMINMEI` | Inflation rate | Monthly | ⭐⭐⭐⭐ |
| Real M2 | Calculated | M2 adjusted for inflation | Monthly | ⭐⭐⭐⭐ |
| Real M2 YoY | Calculated | Real money growth | Monthly | ⭐⭐⭐⭐ |

**Why It Matters:**
- **Japan targeting 2% inflation** - first time in decades
- **Real liquidity** = Nominal liquidity - Inflation
- If inflation exceeds M2 growth, **real liquidity contracts**

### Economic Activity
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| Industrial Production | `JPNPROINDMISMEI` | Manufacturing output | Monthly | ⭐⭐⭐ |
| Retail Sales | `SLRTTO01JPM661S` | Consumer spending | Monthly | ⭐⭐⭐ |

---

## 🌍 GLOBAL CROSS-BORDER INDICATORS (Enhanced)

### USD Funding Markets
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| 3M USD LIBOR | `USD3MTD156N` | Dollar funding cost | Daily | ⭐⭐⭐ |
| SOFR | `SOFR` | Secured overnight rate | Daily | ⭐⭐⭐⭐⭐ |
| Fed Swap Lines | `SWPT` | USD to foreign CBs | Weekly | ⭐⭐⭐⭐⭐ |
| TED Spread | `TEDRATE` | Interbank stress | Daily | ⭐⭐⭐⭐ |
| Eurodollar Deposits | `EURODOL` | Offshore USD | Monthly | ⭐⭐⭐⭐ |

**Key Concept: USD Funding Stress**
- **TED > 50 bps**: Moderate stress
- **TED > 100 bps**: Severe stress (2008, 2020)
- **Fed swaps expanding**: Central banks need USD liquidity

### Credit Spreads (Inverted)
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| HY Spread | `BAMLH0A0HYM2` | Junk bonds vs Treasuries | Daily | ⭐⭐⭐⭐⭐ |
| IG Spread | `BAMLC0A0CM` | Investment grade spread | Daily | ⭐⭐⭐⭐ |
| EM Spread | `BAMLEMRECRPIOAS` | Emerging market spread | Daily | ⭐⭐⭐⭐⭐ |

**Inverted for Liquidity:**
- Narrower spreads = More liquidity = Higher factor
- We invert these so rising spread → falling liquidity factor

### Risk Appetite
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| VIX (Inverted) | `VIXCLS` | Volatility index | Daily | ⭐⭐⭐⭐⭐ |
| Dollar Index | `DTWEXBGS` | DXY strength | Daily | ⭐⭐⭐⭐ |

**Dollar Strength & Liquidity:**
- **DXY rising**: Tighter global USD liquidity
- **DXY falling**: Easier global USD liquidity
- EM assets highly sensitive to DXY

---

## 🏛️ OECD BACKUP INDICATORS

### G7 Aggregates
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| G7 M1 | `MABMM101G7M189S` | Narrow money | Monthly | ⭐⭐⭐ |
| G7 Credit | `QUSM770A` | Credit aggregates | Quarterly | ⭐⭐⭐ |

### Additional Economies
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| UK M4 | `MABMM401GBM189S` | UK broad money | Monthly | ⭐⭐⭐ |
| Canada M2 | `MYAGM2CAM189N` | Canadian money supply | Monthly | ⭐⭐⭐ |
| Australia M3 | `MABMM301AUM189S` | Australian broad money | Monthly | ⭐⭐⭐ |

### Trade Finance
| Indicator | FRED Code | Description | Frequency | Importance |
|-----------|-----------|-------------|-----------|------------|
| World Trade Volume | `WORLD` | Global trade proxy | Monthly | ⭐⭐⭐ |

---

## 📊 TOTAL INDICATOR COUNT

| Region/Category | Count | Coverage |
|----------------|-------|----------|
| 🇺🇸 United States | 10 | Fed, M2, Credit, TGA, RRP |
| 🇪🇺 Eurozone | 3 | ECB, M2, Growth |
| 🇨🇳 **China (Enhanced)** | **17** | **M1/M2, PBoC, TSF, Credit, FX** |
| 🇯🇵 **Japan (Enhanced)** | **19** | **M2/M3, BoJ, YCC, ETF, Real** |
| 🌍 Global | 10 | USD funding, Spreads, VIX |
| 🏛️ OECD | 6 | Backups, G7, Trade |
| **TOTAL** | **65+** | **Comprehensive global coverage** |

---

## 🎯 Regional Weighting Recommendations

Based on systemic importance to global liquidity:

### For PCA Factor (Auto-weighted)
Let PCA determine weights naturally - it will emphasize:
1. Central bank balance sheets (high variance)
2. M2 growth rates (cyclical)
3. Credit spreads (crisis indicators)

### For Custom Weighting
```python
custom_weights = {
    # G3 Central Banks (45%)
    'US_FED_ASSETS': 0.20,
    'EZ_ECB_ASSETS': 0.10,
    'JP_BOJ_ASSETS': 0.15,
    
    # China (20%) - systemic importance
    'CN_M2_YoY': 0.08,
    'CN_PBOC_ASSETS': 0.07,
    'CN_CREDIT_GROWTH': 0.05,
    
    # USD Funding (15%)
    'GLOBAL_SOFR': 0.05,
    'GLOBAL_TED_SPREAD_INV': 0.05,
    'GLOBAL_FED_SWAP': 0.05,
    
    # Credit Markets (15%)
    'GLOBAL_HY_SPREAD_INV': 0.08,
    'GLOBAL_EM_SPREAD_INV': 0.07,
    
    # Other (5%)
    'US_M2_YoY': 0.03,
    'JP_M2_YoY': 0.02,
}
```

---

## 🔗 Data Source Links

### Official Sources
- **FRED**: https://fred.stlouisfed.org/
- **Bank of Japan**: https://www.stat-search.boj.or.jp/
- **PBoC Statistics**: http://www.pbc.gov.cn/en/3688006/index.html
- **OECD Data**: https://data.oecd.org/
- **BIS Statistics**: https://www.bis.org/statistics/

### Alternative Data Providers
- **Bloomberg**: China Total Social Financing (TSF) - `CNFRSTSF Index`
- **Wind**: China credit data (requires subscription)
- **CEIC**: Comprehensive China/Asia data

---

## 📈 Transmission Mechanisms

### China → Global Markets
1. **Commodity demand**: CN credit expansion → Base metals, oil
2. **EM spillovers**: CN easing → ASEAN, LatAm capital inflows
3. **Manufacturing**: CN liquidity → Supply chain financing
4. **Offshore RMB**: CNH liquidity → Hong Kong, Singapore

### Japan → Global Markets
1. **Carry trades**: JPY weakness → Global risk assets
2. **Swap lines**: BoJ USD swaps → Asian dollar liquidity
3. **Sovereign wealth**: GPIF reallocation → Global equities
4. **Safe haven**: JPY strength → Risk-off deleveraging

### USD → Global Markets
1. **Reserve currency**: Fed easing → All risk assets
2. **Cross-border lending**: Dollar shortage → EM stress
3. **Commodity pricing**: Weak dollar → Higher commodity prices
4. **Crypto**: Liquidity surplus → Bitcoin, ETH flows

---

**Last Updated**: February 2026  
**Total Indicators**: 65+  
**Coverage**: G7 + China + Major EMs + Global aggregates
