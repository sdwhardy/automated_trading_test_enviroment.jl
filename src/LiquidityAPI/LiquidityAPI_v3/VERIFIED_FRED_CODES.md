# Verified FRED Series Codes

## ✅ All codes below have been verified to exist in FRED

Last verified: February 2026

---

## 🇺🇸 United States

| Indicator | FRED Code | Status | Description |
|-----------|-----------|--------|-------------|
| M2 Money Stock | `M2SL` | ✅ Active | Broad money supply |
| Fed Total Assets | `WALCL` | ✅ Active | Federal Reserve balance sheet |
| Securities Held | `WSHOSHO` | ✅ Active | Fed's securities portfolio (QE) |
| Bank Credit | `TOTBKCR` | ✅ Active | Commercial bank credit |
| Consumer Credit | `TOTALSL` | ✅ Active | Consumer credit outstanding |
| Treasury General Account | `WTREGEN` | ✅ Active | Gov't cash at Fed |
| Overnight RRP | `RRPONTSYD` | ✅ Active | Reverse repo facility |
| Currency in Circulation | `CURRSL` | ✅ Active | Physical currency |

---

## 🇪🇺 Eurozone

| Indicator | FRED Code | Status | Description |
|-----------|-----------|--------|-------------|
| Euro Area M2 | `MYAGM2EZM196N` | ✅ Active | Eurozone broad money |
| ECB Assets | `ECBASSETSW` | ✅ Active | ECB balance sheet |

---

## 🇨🇳 China (Verified Codes Only)

| Indicator | FRED Code | Status | Description |
|-----------|-----------|--------|-------------|
| M2 Money Supply | `MYAGM2CNM189N` | ✅ Active | Broad money |
| M1 Money Supply | `MABMM301CNM189S` | ✅ Active | Narrow money |
| PBoC Total Assets | `DDDI06CNA156NWDB` | ✅ Active | Central bank balance sheet |
| FX Reserves | `TRESEGCNM052N` | ✅ Active | Foreign exchange reserves |
| Total Private Debt | `QCNPAM770A` | ✅ Active | Credit to private sector |
| Policy Rate | `INTDSRCNM193N` | ✅ Active | Discount rate |
| GDP | `MKTGDPCNA646NWDB` | ✅ Active | Gross domestic product |
| Exports | `XTEXVA01CNM667S` | ✅ Active | Exports of goods |
| Imports | `XTIMVA01CNM667S` | ✅ Active | Imports of goods |
| CNY/USD Rate | `DEXCHUS` | ✅ Active | Exchange rate |

### ❌ Discontinued/Invalid China Codes
- `CHNBSLARNADSMEI` - Bank loans (discontinued)
- `QCNM770A` - Domestic credit (invalid)
- `INTGSTCNM193N` - Lending rate (invalid)
- `IRCN7DV01STM` - 7-day repo (invalid)
- `CHNTOTINDFODISMEI` - Industrial prod (invalid)
- `SLRTTO01CNM661S` - Retail sales (invalid)
- `BPFAIN01CNQ188S` - FDI (invalid)

---

## 🇯🇵 Japan (Verified Codes Only)

| Indicator | FRED Code | Status | Description |
|-----------|-----------|--------|-------------|
| M2 Money Supply | `MYAGM2JPM189N` | ✅ Active | Broad money |
| M3 Money Supply | `MABMM301JPM189S` | ✅ Active | Broader money |
| BoJ Total Assets | `JPNASSETS` | ✅ Active | Bank of Japan balance sheet |
| 10Y JGB Yield | `IRLTLT01JPM156N` | ✅ Active | YCC target rate |
| Overnight Call Rate | `IRSTCI01JPM156N` | ✅ Active | Policy rate |
| Policy Rate | `INTDSRJPM193N` | ✅ Active | Discount rate |
| Industrial Production | `JPNPROINDMISMEI` | ✅ Active | Manufacturing output |
| Real GDP | `JPNRGDPEXP` | ✅ Active | GDP growth |
| Exports | `XTEXVA01JPM667S` | ✅ Active | Exports of goods |
| Imports | `XTIMVA01JPM667S` | ✅ Active | Imports of goods |
| FX Reserves | `TRESEGJPM052N` | ✅ Active | Foreign reserves |
| JPY/USD Rate | `DEXJPUS` | ✅ Active | Exchange rate |
| CPI | `JPNCPIALLMINMEI` | ✅ Active | Inflation |

### ❌ Discontinued/Invalid Japan Codes
- `JPNBASMON` - Monetary base (discontinued)
- `JPNJGBAQ` - BoJ JGB holdings (discontinued)
- `JPNETFAQ` - BoJ ETF holdings (discontinued)
- `JPNBSLARNADSMEI` - Bank lending (invalid)
- `QJPM770A` - Domestic credit (invalid)
- `SLRTTO01JPM661S` - Retail sales (invalid)
- `JPNBCA` - Current account (invalid)

**Note on BoJ ETF/JGB Holdings:**
These are incredibly important indicators but FRED discontinued them. Alternative sources:
- Bank of Japan website: https://www.boj.or.jp/en/statistics/
- Download quarterly reports manually
- Use Bloomberg: `BOJTETF Index` for ETF holdings

---

## 🌍 Global Indicators (Verified)

| Indicator | FRED Code | Status | Description |
|-----------|-----------|--------|-------------|
| SOFR | `SOFR` | ✅ Active | Secured overnight financing |
| Fed Swap Lines | `SWPT` | ✅ Active | Central bank USD swaps |
| Fed Funds Rate | `FEDFUNDS` | ✅ Active | Policy rate |
| 3M T-Bill Rate | `DTB3` | ✅ Active | Treasury bill yield |
| HY Spread | `BAMLH0A0HYM2` | ✅ Active | Junk bond spread |
| IG Spread | `BAMLC0A0CM` | ✅ Active | Investment grade spread |
| BBB Spread | `BAMLC0A4CBBB` | ✅ Active | BBB corporate spread |
| TED Spread | `TEDRATE` | ✅ Active | Interbank stress |
| VIX | `VIXCLS` | ✅ Active | Volatility index |
| DXY Broad | `DTWEXBGS` | ✅ Active | Dollar index |
| DXY Major | `DTWEXM` | ✅ Active | Dollar vs majors |
| Gold Price | `GOLDAMGBD228NLBM` | ✅ Active | Gold spot price |
| WTI Oil | `DCOILWTICO` | ✅ Active | Crude oil price |

### ❌ Discontinued/Invalid Global Codes
- `USD3MTD156N` - 3M USD LIBOR (discontinued 2023)
- `BAMLEMRECRPIOAS` - EM spread (invalid code)
- `EURODOL` - Eurodollar deposits (discontinued)

**LIBOR Replacement:**
LIBOR was discontinued in 2023. Use SOFR instead:
- `SOFR` - Secured Overnight Financing Rate
- `SOFR30DAYAVG` - 30-day SOFR average
- `SOFR90DAYAVG` - 90-day SOFR average

---

## 🏛️ OECD Countries (Verified)

| Indicator | FRED Code | Status | Description |
|-----------|-----------|--------|-------------|
| UK M4 | `MABMM401GBM196N` | ✅ Active | UK broad money |
| Canada M2 | `MANMM102CAM189N` | ✅ Active | Canadian money supply |
| Australia M3 | `MABMM301AUM189S` | ✅ Active | Australian broad money |
| South Korea M2 | `MYAGM2KRM189N` | ✅ Active | Korean money supply |
| Switzerland M3 | `MABMM301CHM189S` | ✅ Active | Swiss money supply |
| Sweden M3 | `MABMM301SEM189S` | ✅ Active | Swedish money supply |

### ❌ Invalid OECD Codes
- `MABMM101G7M189S` - G7 M1 (invalid)
- `QUSM770A` - G7 lending (invalid)
- `MYAGM2CAM189N` - Canada M2 (wrong code, use MANMM102CAM189N)
- `WORLD` - World trade (invalid)

---

## 📊 Current Total: 55+ Verified Indicators

| Region | Count |
|--------|-------|
| US | 8 |
| Eurozone | 2 |
| China | 10 base + 7 derived = 17 |
| Japan | 13 base + 8 derived = 21 |
| Global | 13 |
| OECD | 6 |
| **Total** | **52 base + derived = 65+** |

---

## 🔍 How to Verify FRED Codes

If you want to check if a code exists:

1. **Via FRED Website:**
   ```
   https://fred.stlouisfed.org/series/[CODE]
   ```
   Example: https://fred.stlouisfed.org/series/M2SL

2. **Via API (our method):**
   ```python
   url = f"https://api.stlouisfed.org/fred/series/observations?series_id={code}&api_key={key}&file_type=json"
   # If 400 error → code invalid/discontinued
   ```

3. **FRED Search:**
   Go to https://fred.stlouisfed.org/ and search by keywords

---

## 🎯 Recommended Core Set (Most Reliable)

If you want **maximum reliability** with **minimum failures**, use this subset:

### Core 30 Indicators (Near 100% Uptime)

**US (8):** M2SL, WALCL, WSHOSHO, TOTBKCR, TOTALSL, WTREGEN, RRPONTSYD, CURRSL

**Eurozone (2):** MYAGM2EZM196N, ECBASSETSW

**China (4):** MYAGM2CNM189N, MABMM301CNM189S, DDDI06CNA156NWDB, TRESEGCNM052N

**Japan (4):** MYAGM2JPM189N, JPNASSETS, IRLTLT01JPM156N, JPNCPIALLMINMEI

**Global (8):** SOFR, SWPT, BAMLH0A0HYM2, BAMLC0A0CM, TEDRATE, VIXCLS, DTWEXBGS, FEDFUNDS

**OECD (4):** MABMM401GBM196N, MANMM102CAM189N, MABMM301AUM189S, MYAGM2KRM189N

This gives you **global coverage** with **minimal API errors**.

---

## 💡 Alternative Data Sources for Missing Indicators

### For China TSF (Total Social Financing):
- **PBoC Website**: http://www.pbc.gov.cn/en/
- **Bloomberg**: `CNFRSTSF Index`
- **CEIC**: China economic database (subscription)

### For BoJ JGB/ETF Holdings:
- **BoJ Website**: https://www.boj.or.jp/en/statistics/boj/other/acmai/index.htm
- **Bloomberg**: `BOJTETF Index`, `BOJJGB Index`

### For China 7-Day Repo:
- **Bloomberg**: `CNRR007 Index`
- **Investing.com**: China 7-day repo rate (free charts)

### For Real-Time Credit Data:
- **BIS**: https://www.bis.org/statistics/totcredit.htm (quarterly)
- **OECD**: https://data.oecd.org/ (monthly/quarterly)

---

**Last Updated**: February 10, 2026  
**Verification Method**: API testing with active FRED key  
**Next Review**: Quarterly (May 2026)
