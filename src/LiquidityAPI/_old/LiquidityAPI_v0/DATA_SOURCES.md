# Data Sources and Indicator Guide

## Complete List of Indicators

### United States Indicators (FRED)

#### Monetary Aggregates
| Indicator | FRED Code | Description | Frequency | Units |
|-----------|-----------|-------------|-----------|-------|
| M2 Money Stock | `M2SL` | Broad money supply including cash, checking deposits, savings deposits | Monthly | Billions USD |
| M2 YoY Growth | Calculated | Year-over-year percentage change in M2 | Monthly | Percent |
| Currency in Circulation | `CURRSL` | Physical currency outside Treasury and Fed | Monthly | Billions USD |

#### Federal Reserve Balance Sheet
| Indicator | FRED Code | Description | Frequency | Units |
|-----------|-----------|-------------|-----------|-------|
| Fed Total Assets | `WALCL` | All assets on Federal Reserve balance sheet | Weekly | Billions USD |
| Securities Held Outright | `WSHOSHO` | Treasury and MBS holdings (QE indicator) | Weekly | Billions USD |
| Overnight Reverse Repo | `RRPONTSYD` | Fed's liquidity absorption facility | Daily | Billions USD |
| Treasury General Account | `WTREGEN` | Government's checking account at Fed (reverse liquidity) | Weekly | Billions USD |
| Fed Liquidity Swaps | `SWPT` | USD provided to foreign central banks | Weekly | Billions USD |

#### Credit Indicators
| Indicator | FRED Code | Description | Frequency | Units |
|-----------|-----------|-------------|-----------|-------|
| Bank Credit | `TOTBKCR` | Total credit extended by commercial banks | Weekly | Billions USD |
| Consumer Credit | `TOTALSL` | Household credit outstanding | Monthly | Billions USD |

### Eurozone Indicators (ECB/FRED)

| Indicator | FRED Code | Description | Frequency | Units |
|-----------|-----------|-------------|-----------|-------|
| Euro Area M2 | `MYAGM2EZM196N` | Eurozone broad money supply | Monthly | Billions EUR |
| ECB Assets | `ECBASSETSW` | European Central Bank balance sheet | Weekly | Billions EUR |

### China Indicators (FRED)

| Indicator | FRED Code | Description | Frequency | Units |
|-----------|-----------|-------------|-----------|-------|
| China M2 | `MYAGM2CNM189N` | Chinese money supply | Monthly | Billions CNY |
| PBoC Assets | `DDDI06CNA156NWDB` | People's Bank of China balance sheet | Monthly | Billions CNY |

### Japan Indicators (FRED)

| Indicator | FRED Code | Description | Frequency | Units |
|-----------|-----------|-------------|-----------|-------|
| Japan M2 | `MYAGM2JPM189N` | Japanese money supply | Monthly | Billions JPY |
| BoJ Assets | `JPNASSETS` | Bank of Japan balance sheet | Monthly | Billions JPY |

### Global Cross-Border Indicators (FRED/BIS)

#### USD Funding Markets
| Indicator | FRED Code | Description | Frequency | Units |
|-----------|-----------|-------------|-----------|-------|
| 3-Month USD LIBOR | `USD3MTD156N` | Key USD funding rate (discontinued 2023) | Daily | Percent |
| TED Spread | `TEDRATE` | 3M LIBOR - 3M T-bill (interbank stress) | Daily | Percentage Points |

#### Credit Spreads
| Indicator | FRED Code | Description | Frequency | Units |
|-----------|-----------|-------------|-----------|-------|
| High Yield Spread | `BAMLH0A0HYM2` | HY corporate bonds vs Treasuries | Daily | Percentage Points |
| Investment Grade Spread | `BAMLC0A0CM` | IG corporate bonds vs Treasuries | Daily | Percentage Points |

#### Risk Indicators
| Indicator | FRED Code | Description | Frequency | Units |
|-----------|-----------|-------------|-----------|-------|
| VIX | `VIXCLS` | CBOE Volatility Index (fear gauge) | Daily | Index Points |

---

## Data Source APIs

### 1. FRED (Federal Reserve Economic Data)

**Base URL**: `https://api.stlouisfed.org/fred/series/observations`

**Authentication**: API key required (free)
- Sign up: https://fred.stlouisfed.org/docs/api/api_key.html
- Include in query params: `api_key=YOUR_KEY`

**Request Format**:
```
GET https://api.stlouisfed.org/fred/series/observations?
    series_id=M2SL&
    api_key=YOUR_KEY&
    file_type=json&
    observation_start=2010-01-01&
    observation_end=2024-12-31
```

**Response Format**:
```json
{
  "observations": [
    {
      "date": "2010-01-01",
      "value": "8532.9"
    }
  ]
}
```

**Rate Limits**: No strict limit for registered users
**Documentation**: https://fred.stlouisfed.org/docs/api/fred/

---

### 2. ECB SDMX API (European Central Bank)

**Base URL**: `https://data-api.ecb.europa.eu/service/data`

**Authentication**: None required

**Request Format**:
```
GET https://data-api.ecb.europa.eu/service/data/BSI/M.U2.N.A.A20.A.1.U2.2300.Z01.E?
    startPeriod=2010-01&
    endPeriod=2024-12&
    format=jsondata
```

**Dataset Codes**:
- `BSI` - Balance Sheet Items (monetary aggregates, bank balance sheets)
- `QSA` - Euro area accounts (quarterly financial accounts)
- `ICP` - Insurance corporation and pension funds
- `IVF` - Investment funds

**Key Dimensions**:
- `M` = Monthly frequency
- `U2` = Euro area
- `A20` = M2 monetary aggregate

**Documentation**: https://data.ecb.europa.eu/help/api/overview

---

### 3. BIS API (Bank for International Settlements)

**Base URL**: `https://data.bis.org/api/v1`

**Authentication**: None required

**Request Format**:
```
GET https://data.bis.org/api/v1/data/WEBSTATS_LBS_D_PUB/Q.S.A.A.A.5J.A.5A.N?
    start_period=2010-Q1&
    end_period=2024-Q4&
    format=json
```

**Key Datasets**:
- `WEBSTATS_LBS_D_PUB` - Locational banking statistics (cross-border credit)
- `WEBSTATS_CBS_PUB` - Consolidated banking statistics
- `WEBSTATS_CREDIT_PUB` - Credit to non-financial sector
- `WEBSTATS_DEBTSEC` - Debt securities statistics

**Documentation**: https://www.bis.org/statistics/api_documentation.htm

---

### 4. IMF API (International Monetary Fund)

**Base URL**: `http://dataservices.imf.org/REST/SDMX_JSON.svc`

**Authentication**: None required

**Request Format**:
```
GET http://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS/M.US.FMANBM_BP6_USD
```

**Key Databases**:
- `IFS` - International Financial Statistics (monetary data, balance of payments)
- `BOP` - Balance of Payments and International Investment Position
- `GFSR` - Global Financial Stability Report
- `DOT` - Direction of Trade Statistics

**Country Codes**:
- `US` - United States
- `CN` - China
- `JP` - Japan
- `U2` - Euro Area

**Indicator Categories**:
- `FM*` - Monetary aggregates (M1, M2, M3)
- `FC*` - Credit indicators
- `FI*` - Interest rates

**Documentation**: https://datahelp.imf.org/knowledgebase/articles/667681

---

## Additional Potential Data Sources

### World Bank Open Data
- **URL**: https://data.worldbank.org/
- **API**: https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
- **Use case**: Long-term historical macroeconomic data

### OECD Data
- **URL**: https://data.oecd.org/
- **API**: https://data.oecd.org/api/
- **Use case**: Cross-country comparable indicators

### Bloomberg API (Paid)
- **Use case**: High-frequency market data, proprietary liquidity indicators
- **Indicators**: WIRP (World Interest Rate Probabilities), MLIV (Markets Live)

### Refinitiv/LSEG Data (Paid)
- **Use case**: Institutional liquidity flows, FX swaps, repo markets

---

## Indicator Selection Rationale

### Why These Indicators?

**Central Bank Balance Sheets (Weight: High)**
- Direct measure of liquidity injection/withdrawal
- QE/QT programs show up clearly
- Weekly frequency for timely signals

**Monetary Aggregates (Weight: Medium-High)**
- Captures broad money creation
- Includes commercial bank lending multiplier
- Monthly frequency, available globally

**Credit Growth (Weight: Medium)**
- Shows private sector liquidity creation
- Leading indicator for economic activity
- Complementary to central bank actions

**USD Funding Markets (Weight: High)**
- USD is global reserve currency
- Cross-border credit impacts all markets
- Captures offshore dollar demand

**Credit Spreads (Weight: Medium)**
- Inverse indicator (tight = more liquidity)
- Market-based vs policy-based measure
- High frequency (daily)

**Risk Indicators (Weight: Low-Medium)**
- VIX captures market liquidity demand
- Complements supply-side measures
- Can signal regime changes

---

## Data Quality Notes

### Reliability by Source
1. **FRED** (★★★★★) - Most reliable, well-maintained, rarely revised
2. **ECB** (★★★★☆) - Reliable but occasional delays
3. **BIS** (★★★★☆) - Quarterly lag, but authoritative
4. **IMF** (★★★☆☆) - Can have substantial delays

### Update Frequencies
- **Daily**: Credit spreads, VIX, reverse repo
- **Weekly**: Fed balance sheet, bank credit
- **Monthly**: M2, consumer credit, China/Japan/EZ data
- **Quarterly**: BIS cross-border data

### Historical Availability
- **FRED**: Most series from 1950s-present
- **ECB**: Euro era (1999-present)
- **BIS**: 1970s-present (varies by series)
- **IMF**: 1990s-present (varies by country)

---

## Alternative Indicators (Not Included but Recommended)

### For Future Enhancement

**UK / Switzerland**
- Bank of England balance sheet (weekly)
- Swiss National Bank sight deposits (indicator of FX interventions)

**Crypto On-Chain Metrics**
- Stablecoin supply (USDT, USDC on-chain data)
- Exchange reserves (liquidity available for crypto trading)
- Bitcoin difficulty / hash rate

**Repo Markets**
- SOFR (Secured Overnight Financing Rate) - replaced LIBOR
- GCF Repo rates (general collateral financing)
- DTCC tri-party repo data

**Shadow Banking**
- Money market fund assets (FRED: MMMFFAQ027S)
- Hedge fund leverage (CFTC data)
- Prime brokerage data (if available)

**Real-Time Proxies**
- Google Trends for "liquidity crisis", "cash", etc.
- Twitter sentiment on central bank actions
- High-frequency credit card spending data

---

## Updating This Guide

As financial markets evolve, new liquidity indicators emerge. Key events to monitor:

1. **LIBOR Transition** (2021-2023) - replaced by SOFR/SONIA
2. **Central Bank Digital Currencies** - new liquidity channels
3. **Stablecoin Regulations** - could affect crypto liquidity transmission
4. **Climate Finance** - green QE programs

---

## References

- Federal Reserve Board. (2024). *Federal Reserve Statistical Release*. https://www.federalreserve.gov/
- European Central Bank. (2024). *Statistical Data Warehouse*. https://sdw.ecb.europa.eu/
- Bank for International Settlements. (2024). *BIS Statistics*. https://www.bis.org/statistics/
- International Monetary Fund. (2024). *IMF Data*. https://data.imf.org/

---

**Last Updated**: February 2026
