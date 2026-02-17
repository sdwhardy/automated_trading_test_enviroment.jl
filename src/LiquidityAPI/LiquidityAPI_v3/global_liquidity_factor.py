"""
Global Liquidity Factor — Block PCA
=====================================
Constructs 6 independent liquidity scores by running PCA within 5 thematic
blocks, each capturing a distinct dimension of global liquidity:

  Block                  PCs   Anchor indicator (higher → more liquidity)
  ─────────────────────────────────────────────────────────────────────────
  Monetary Aggregates     2    US_M2
  CB Balance Sheets       1    US_FED_ASSETS
  Credit Spreads          1    GLOBAL_HY_SPREAD_INV  (inverted OAS)
  FX / Commodities        1    GLOBAL_DXY_INV        (inverted dollar)
  Macro Flows             1    CN_EXPORTS

PC1 of every block is auto-oriented so that HIGHER score = MORE liquidity /
more risk-on, using the anchor indicator defined per block.

Data sources : FRED (primary), ECB SDMX, BIS, IMF
Frequency    : Month-end (ME), z-score normalised, rolling 36-month window
Outputs      : results/block_pca_scores.csv   — 6-column time series
               results/block_pca_loadings.csv — loadings + explained variance
               results/block_pca_scores.png   — 5-panel chart
               results/block_pca_correlation.png
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for data sources and API endpoints"""
    
    # FRED API
    FRED_API_KEY = "6dd5d65c0b6778e9433a7934ee82eb94"  # Get from https://fred.stlouisfed.org/docs/api/api_key.html
    FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"
    
    # ECB SDMX API
    ECB_BASE_URL = "https://data-api.ecb.europa.eu/service/data"
    
    # BIS API
    BIS_BASE_URL = "https://data.bis.org/api/v1"
    
    # IMF SDMX API
    IMF_BASE_URL = "http://dataservices.imf.org/REST/SDMX_JSON.svc"
    
    # Bank of Japan API
    BOJ_BASE_URL = "https://www.stat-search.boj.or.jp/ssi/cgi-bin/famecgi2"
    
    # OECD API
    OECD_BASE_URL = "https://sdmx.oecd.org/public/rest/data"
    
    # Date range
    START_DATE = "2010-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    
    # Target frequency for standardization
    TARGET_FREQ = "ME"  # 'W' for weekly, 'ME' for month-end (pandas 2.2+)


# ============================================================================
# DATA FETCHERS
# ============================================================================

class FREDFetcher:
    """Fetch data from FRED API"""

    # FRED blocks these sources from API redistribution → always 400
    # src: https://fred.stlouisfed.org/docs/api/fred/ + fredr package docs
    API_BLOCKED = {
        # LBMA precious metals (never redistributable via API)
        'GOLDAMGBD228NLBM': 'LBMA Gold AM – not redistributable via API',
        'GOLDPMGBD228NLBM': 'LBMA Gold PM – not redistributable via API',
        'SLVPRUSD':         'LBMA Silver – not redistributable via API',
        # ICE benchmarks
        'USD3MTD156N':      'ICE USD LIBOR – discontinued 2023; use SOFR',
        'EUR3MTD156N':      'ICE EUR LIBOR – discontinued',
        'GBP3MTD156N':      'ICE GBP LIBOR – discontinued',
        # Codes that simply do not exist on FRED
        'MOVE':             'Not on FRED; use EMVOVERALLEMV instead',
        'DBMTXBMIDINDXM':   'Not on FRED; Baltic Dry not available via FRED API',
        'BAMLHE00EHY0Y':    'Wrong code; correct Euro HY OAS is BAMLHE00EHYIOAS',
        'MABMM401GBM196N':  'Wrong code; correct UK M4 is MYAGM4GBM189N',
        'MANMM102CAM189N':  'Wrong code; correct Canada broad money is MABMM301CAM189S',
        'MABMM101G7M189S':  'Not on FRED',
        'QUSM770A':         'Not on FRED',
        'WORLD':            'Not on FRED',
        'BAMLEMRECRPIOAS':  'Wrong code; no direct EM spread available freely on FRED',
        'EURODOL':          'Discontinued',
        # CBOE (may be blocked depending on key/region)
        'VIXCLS':           'CBOE VIX – use EMVOVERALLEMV as public-domain alternative',
    }

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = Config.FRED_BASE_URL

    def fetch_series(self, series_id, start_date, end_date):
        """Fetch a single FRED series, with early-exit for known-blocked codes."""

        if series_id in self.API_BLOCKED:
            print(f"  ⚠  SKIPPED {series_id}: {self.API_BLOCKED[series_id]}")
            return pd.DataFrame()

        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            if response.status_code == 400:
                # Surface the actual FRED error message for easier debugging
                try:
                    msg = response.json().get('error_message', response.text[:120])
                except Exception:
                    msg = response.text[:120]
                print(f"  ✗  {series_id} – FRED 400: {msg}")
                return pd.DataFrame()
            response.raise_for_status()
            data = response.json()

            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df[['date', 'value']].set_index('date')
                df.columns = [series_id]
                return df
        except Exception as e:
            print(f"  ✗  Error fetching {series_id}: {e}")
        return pd.DataFrame()

    def fetch_multiple(self, series_dict, start_date, end_date):
        """Fetch multiple FRED series, skipping any that return no data."""
        dfs = []
        for name, series_id in series_dict.items():
            print(f"Fetching {name} ({series_id})...")
            df = self.fetch_series(series_id, start_date, end_date)
            if not df.empty:
                df.columns = [name]
                dfs.append(df)

        if dfs:
            return pd.concat(dfs, axis=1)
        return pd.DataFrame()


class ECBFetcher:
    """Fetch data from ECB SDMX API"""
    
    def __init__(self):
        self.base_url = Config.ECB_BASE_URL
    
    def fetch_series(self, flow, key, start_date, end_date):
        """
        Fetch ECB data via SDMX
        flow: dataset code (e.g., 'BSI' for Balance Sheet Items)
        key: series key
        """
        url = f"{self.base_url}/{flow}/{key}"
        params = {
            'startPeriod': start_date,
            'endPeriod': end_date,
            'format': 'jsondata'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse SDMX-JSON structure
            if 'dataSets' in data and len(data['dataSets']) > 0:
                observations = data['dataSets'][0].get('series', {})
                time_periods = data['structure']['dimensions']['observation'][0]['values']
                
                dates = [period['id'] for period in time_periods]
                values = []
                
                for series_key, series_data in observations.items():
                    obs = series_data.get('observations', {})
                    values = [float(obs[str(i)][0]) if str(i) in obs else np.nan 
                             for i in range(len(dates))]
                    break  # Take first series
                
                df = pd.DataFrame({'date': dates, 'value': values})
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                return df
        except Exception as e:
            print(f"Error fetching ECB data: {str(e)}")
            return pd.DataFrame()


class BISFetcher:
    """Fetch data from BIS API"""
    
    def __init__(self):
        self.base_url = Config.BIS_BASE_URL
    
    def fetch_series(self, series_id, start_date, end_date):
        """Fetch BIS series"""
        url = f"{self.base_url}/data/{series_id}"
        params = {
            'start_period': start_date,
            'end_period': end_date,
            'format': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'observations' in data:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['period'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df[['date', 'value']].set_index('date')
                return df
        except Exception as e:
            print(f"Error fetching BIS data {series_id}: {str(e)}")
            return pd.DataFrame()


class IMFFetcher:
    """Fetch data from IMF API"""
    
    def __init__(self):
        self.base_url = Config.IMF_BASE_URL
    
    def fetch_series(self, database, country, indicator, start_year):
        """
        Fetch IMF data
        database: e.g., 'IFS' (International Financial Statistics)
        country: country code
        indicator: indicator code
        """
        url = f"{self.base_url}/CompactData/{database}/M.{country}.{indicator}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse IMF JSON structure
            observations = data.get('CompactData', {}).get('DataSet', {}).get('Series', {}).get('Obs', [])
            
            dates = []
            values = []
            for obs in observations:
                if '@TIME_PERIOD' in obs and '@OBS_VALUE' in obs:
                    dates.append(obs['@TIME_PERIOD'])
                    values.append(float(obs['@OBS_VALUE']))
            
            df = pd.DataFrame({'date': dates, 'value': values})
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
        except Exception as e:
            print(f"Error fetching IMF data: {str(e)}")
            return pd.DataFrame()


class BOJFetcher:
    """Fetch data from Bank of Japan"""
    
    def __init__(self):
        self.base_url = Config.BOJ_BASE_URL
    
    def fetch_boj_data(self):
        """
        Fetch key BOJ indicators
        Note: BOJ API is complex. For production, use their CSV downloads or FRED proxies.
        This fetcher provides framework for direct BOJ integration.
        """
        print("Note: Using FRED proxies for BOJ data. Direct BOJ API integration available via CSV downloads.")
        return pd.DataFrame()


class OECDFetcher:
    """Fetch data from OECD SDMX API"""
    
    def __init__(self):
        self.base_url = Config.OECD_BASE_URL
    
    def fetch_series(self, dataset, key, start_date, end_date):
        """
        Fetch OECD data via SDMX
        dataset: e.g., 'MEI' (Main Economic Indicators), 'QNA' (Quarterly National Accounts)
        key: series key (e.g., 'JPN.M2.GP.M')
        """
        url = f"{self.base_url}/{dataset}/{key}"
        params = {
            'startPeriod': start_date[:7],  # YYYY-MM format
            'endPeriod': end_date[:7],
            'format': 'json'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Parse OECD SDMX-JSON structure
            if 'dataSets' in data and len(data['dataSets']) > 0:
                dataset_info = data['dataSets'][0]
                
                # Extract observations
                if 'series' in dataset_info:
                    for series_key, series_data in dataset_info['series'].items():
                        observations = series_data.get('observations', {})
                        
                        # Get time dimension
                        time_periods = data['structure']['dimensions']['observation'][0]['values']
                        
                        dates = []
                        values = []
                        for idx, period_info in enumerate(time_periods):
                            if str(idx) in observations:
                                dates.append(period_info['id'])
                                values.append(float(observations[str(idx)][0]))
                        
                        df = pd.DataFrame({'date': dates, 'value': values})
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                        return df
            
            return pd.DataFrame()
        except Exception as e:
            print(f"Error fetching OECD data: {str(e)}")
            return pd.DataFrame()


# ============================================================================
# LIQUIDITY FACTOR BUILDER
# ============================================================================

class GlobalLiquidityFactor:
    """Build and manage global liquidity factor"""
    
    def __init__(self, fred_api_key):
        self.fred = FREDFetcher(fred_api_key)
        self.ecb = ECBFetcher()
        self.bis = BISFetcher()
        self.imf = IMFFetcher()
        self.boj = BOJFetcher()
        self.oecd = OECDFetcher()
        self.data = pd.DataFrame()
        self.factor = pd.Series()
    
    def fetch_us_liquidity(self):
        """Fetch US liquidity indicators from FRED"""
        print("\n=== Fetching US Liquidity Data ===")
        
        us_series = {
            # Monetary aggregates
            'US_M2': 'M2SL',                    # M2 Money Stock
            'US_M2_YoY': 'M2SL',                # Will calculate YoY
            
            # Federal Reserve Balance Sheet
            'US_FED_ASSETS': 'WALCL',           # Total Assets
            'US_FED_SECURITIES': 'WSHOSHO',     # Securities Held Outright
            
            # Credit
            'US_BANK_CREDIT': 'TOTBKCR',        # Bank Credit All Commercial Banks
            'US_CONSUMER_CREDIT': 'TOTALSL',    # Total Consumer Credit
            
            # Treasury General Account (reverse liquidity)
            'US_TGA': 'WTREGEN',                # Treasury General Account
            
            # Reverse Repo (liquidity drain)
            'US_RRPONTSYD': 'RRPONTSYD',        # Overnight Reverse Repo
            
            # Currency in circulation
            'US_CURRENCY': 'CURRSL',            # Currency in Circulation
        }
        
        df = self.fred.fetch_multiple(us_series, Config.START_DATE, Config.END_DATE)
        
        # Calculate YoY growth rates
        if 'US_M2' in df.columns:
            df['US_M2_YoY'] = df['US_M2'].pct_change(12) * 100
        
        return df
    
    def fetch_eurozone_liquidity(self):
        """Fetch Eurozone liquidity indicators"""
        print("\n=== Fetching Eurozone Liquidity Data ===")
        
        # Use FRED for easily accessible Eurozone data
        ez_series = {
            'EZ_M2': 'MYAGM2EZM196N',           # M2 for Euro Area
            'EZ_ECB_ASSETS': 'ECBASSETSW',      # ECB Assets
        }
        
        df = self.fred.fetch_multiple(ez_series, Config.START_DATE, Config.END_DATE)
        
        # Calculate YoY growth
        if 'EZ_M2' in df.columns:
            df['EZ_M2_YoY'] = df['EZ_M2'].pct_change(12) * 100
        
        return df
    
    def fetch_china_liquidity(self):
        """Fetch China liquidity indicators - using verified FRED codes"""
        print("\n=== Fetching China Liquidity Data ===")
        
        china_series = {
            # Monetary Aggregates (verified codes)
            'CN_M2': 'MYAGM2CNM189N',                    # M2 Money Supply ✓
            'CN_M1': 'MABMM301CNM189S',                  # M1 Money Supply ✓
            'CN_PBOC_ASSETS': 'DDDI06CNA156NWDB',        # PBoC Total Assets ✓
            'CN_RESERVE_ASSETS': 'TRESEGCNM052N',        # Foreign Exchange Reserves ✓
            
            # Credit & Debt (verified codes)
            'CN_TOTAL_DEBT': 'QCNPAM770A',               # Total Credit to Private Non-Financial Sector ✓
            'CN_HOUSEHOLD_DEBT': 'QCNPAM770A',           # Household debt (part of total)
            
            # Interest Rates (verified codes)
            'CN_POLICY_RATE': 'INTDSRCNM193N',           # Discount Rate (proxy for policy)
            'CN_DEPOSIT_RATE': 'INTDSRCNM193N',          # Deposit Rate
            
            # Economic Activity
            'CN_GDP': 'MKTGDPCNA646NWDB',                # GDP (annual)
            'CN_EXPORTS': 'XTEXVA01CNM667S',             # Exports
            'CN_IMPORTS': 'XTIMVA01CNM667S',             # Imports
            
            # Exchange Rate
            'CN_EXCHANGE_RATE': 'DEXCHUS',               # CNY/USD Exchange Rate
        }
        
        df = self.fred.fetch_multiple(china_series, Config.START_DATE, Config.END_DATE)
        
        # Calculate growth rates for key indicators
        if 'CN_M2' in df.columns:
            df['CN_M2_YoY'] = df['CN_M2'].pct_change(12) * 100
            df['CN_M2_MoM'] = df['CN_M2'].pct_change(1) * 100
        
        if 'CN_M1' in df.columns:
            df['CN_M1_YoY'] = df['CN_M1'].pct_change(12) * 100
        
        if 'CN_TOTAL_DEBT' in df.columns:
            df['CN_DEBT_GROWTH'] = df['CN_TOTAL_DEBT'].pct_change(4) * 100  # Quarterly
        
        # M1/M2 ratio (liquidity velocity indicator)
        if 'CN_M1' in df.columns and 'CN_M2' in df.columns:
            df['CN_M1_M2_RATIO'] = (df['CN_M1'] / df['CN_M2']) * 100
        
        # FX Reserves momentum
        if 'CN_RESERVE_ASSETS' in df.columns:
            df['CN_FX_CHANGE'] = df['CN_RESERVE_ASSETS'].pct_change(12) * 100
        
        # Trade balance (exports - imports)
        if 'CN_EXPORTS' in df.columns and 'CN_IMPORTS' in df.columns:
            df['CN_TRADE_BALANCE'] = df['CN_EXPORTS'] - df['CN_IMPORTS']
        
        print(f"✓ Fetched {len([c for c in df.columns if c.startswith('CN_')])} China indicators")
        
        return df
    
    def fetch_japan_liquidity(self):
        """Fetch Japan liquidity indicators - using verified FRED codes"""
        print("\n=== Fetching Japan Liquidity Data ===")
        
        japan_series = {
            # Monetary Aggregates (verified)
            'JP_M2': 'MYAGM2JPM189N',                    # M2 Money Supply ✓
            'JP_M3': 'MABMM301JPM189S',                  # M3 Money Supply ✓
            'JP_MONETARY_BASE': 'BOGMBASE',              # Monetary Base (US proxy, better availability)
            
            # Bank of Japan Balance Sheet (verified)
            'JP_BOJ_ASSETS': 'JPNASSETS',                # BoJ Total Assets ✓
            
            # Interest Rates (verified)
            'JP_10Y_YIELD': 'IRLTLT01JPM156N',           # 10-Year JGB Yield ✓
            'JP_CALL_RATE': 'IRSTCI01JPM156N',           # Overnight Call Rate ✓
            'JP_POLICY_RATE': 'INTDSRJPM193N',           # Discount Rate
            
            # Economic Activity (verified)
            'JP_INDUSTRIAL_PROD': 'JPNPROINDMISMEI',     # Industrial Production ✓
            'JP_GDP': 'JPNRGDPEXP',                      # Real GDP
            'JP_EXPORTS': 'XTEXVA01JPM667S',             # Exports
            'JP_IMPORTS': 'XTIMVA01JPM667S',             # Imports
            
            # Currency & Reserves (verified)
            'JP_FOREX_RESERVES': 'TRESEGJPM052N',        # FX Reserves ✓
            'JP_EXCHANGE_RATE': 'DEXJPUS',               # JPY/USD Exchange Rate
            
            # Inflation (verified)
            'JP_CPI': 'JPNCPIALLMINMEI',                 # CPI All Items ✓
        }
        
        df = self.fred.fetch_multiple(japan_series, Config.START_DATE, Config.END_DATE)
        
        # Calculate growth rates
        if 'JP_M2' in df.columns:
            df['JP_M2_YoY'] = df['JP_M2'].pct_change(12) * 100
            df['JP_M2_MoM'] = df['JP_M2'].pct_change(1) * 100
        
        if 'JP_M3' in df.columns:
            df['JP_M3_YoY'] = df['JP_M3'].pct_change(12) * 100
        
        if 'JP_BOJ_ASSETS' in df.columns:
            df['JP_BOJ_GROWTH'] = df['JP_BOJ_ASSETS'].pct_change(12) * 100
        
        # Real money supply (adjust for inflation)
        if 'JP_M2' in df.columns and 'JP_CPI' in df.columns:
            # Normalize CPI to base 100
            cpi_normalized = (df['JP_CPI'] / df['JP_CPI'].iloc[0]) * 100
            df['JP_REAL_M2'] = (df['JP_M2'] / cpi_normalized) * 100
            df['JP_REAL_M2_YoY'] = df['JP_REAL_M2'].pct_change(12) * 100
        
        # Trade balance
        if 'JP_EXPORTS' in df.columns and 'JP_IMPORTS' in df.columns:
            df['JP_TRADE_BALANCE'] = df['JP_EXPORTS'] - df['JP_IMPORTS']
        
        # FX reserves momentum
        if 'JP_FOREX_RESERVES' in df.columns:
            df['JP_FX_CHANGE'] = df['JP_FOREX_RESERVES'].pct_change(12) * 100
        
        print(f"✓ Fetched {len([c for c in df.columns if c.startswith('JP_')])} Japan indicators")
        
        return df
    
    def fetch_global_indicators(self):
        """
        Fetch global cross-border and USD funding indicators.

        ⚠️  FRED API redistribution restrictions (returns 400 for these sources):
              • LBMA Gold / Silver  → use EIA oil (DCOILWTICO) as commodity proxy
              • ICE LIBOR / Swap    → replaced by SOFR
              • CBOE VIX            → use EMVOVERALLEMV (public domain) instead
        All codes below are verified Public Domain / Federal Reserve sources.
        """
        print("\n=== Fetching Global Liquidity Indicators ===")

        global_series = {
            # ── USD funding (Fed / Treasury sources → freely available) ──
            'GLOBAL_SOFR':        'SOFR',              # Secured Overnight Financing Rate
            'GLOBAL_FED_FUNDS':   'FEDFUNDS',          # Effective Fed Funds Rate
            'GLOBAL_3M_TBILL':    'DTB3',              # 3-Month Treasury Bill
            'GLOBAL_FED_SWAP':    'SWPT',              # Fed central-bank liquidity swap lines
            'GLOBAL_TED_SPREAD':  'TEDRATE',           # TED Spread

            # ── Credit spreads (ICE BofA – Fed redistributes freely) ──
            'GLOBAL_HY_SPREAD':   'BAMLH0A0HYM2',     # US HY OAS
            'GLOBAL_IG_SPREAD':   'BAMLC0A0CM',       # US IG OAS
            'GLOBAL_BBB_SPREAD':  'BAMLC0A4CBBB',     # US BBB OAS
            'GLOBAL_HY_EU_SPREAD':'BAMLHE00EHYIOAS',  # Euro HY OAS ✓ (confirmed)

            # ── Volatility (public-domain academic series, monthly) ──
            'GLOBAL_EMV':         'EMVOVERALLEMV',    # Equity Market Volatility Tracker ✓

            # ── Dollar indices (Fed Board → freely available) ──
            'GLOBAL_DXY':         'DTWEXBGS',         # Broad Dollar Index
            'GLOBAL_DXY_MAJOR':   'DTWEXM',           # Dollar vs Major Currencies

            # ── Commodity proxies (EIA / World Bank → public domain) ──
            'GLOBAL_OIL_WTI':     'DCOILWTICO',       # WTI Crude (EIA)
            'GLOBAL_OIL_BRENT':   'DCOILBRENTEU',     # Brent Crude (EIA)
            'GLOBAL_COPPER':      'PCOPPUSDM',        # Copper Price (World Bank)
        }

        df = self.fred.fetch_multiple(global_series, Config.START_DATE, Config.END_DATE)

        # Invert: lower spread / lower vol = more liquidity → flip sign so factor rises
        for col in ['GLOBAL_HY_SPREAD', 'GLOBAL_IG_SPREAD', 'GLOBAL_BBB_SPREAD',
                    'GLOBAL_HY_EU_SPREAD', 'GLOBAL_TED_SPREAD', 'GLOBAL_EMV']:
            if col in df.columns:
                df[f"{col}_INV"] = -df[col]

        # Weaker dollar = looser global USD liquidity
        for col in ['GLOBAL_DXY', 'GLOBAL_DXY_MAJOR']:
            if col in df.columns:
                df[f"{col}_INV"] = -df[col]

        # Higher SOFR = tighter, so include level but also inverted for factor use
        if 'GLOBAL_SOFR' in df.columns:
            df['GLOBAL_SOFR_INV'] = -df['GLOBAL_SOFR']

        return df
    
    def fetch_oecd_liquidity(self):
        """Fetch OECD data as backup - verified codes only"""
        print("\n=== Fetching OECD Liquidity Data ===")
        
        # Using only confirmed-active FRED codes (verified Feb 2026)
        oecd_series = {
            'OECD_UK_M4':         'MYAGM4GBM189N',    # UK M4 money stock ✓ (IMF/IFS)
            'OECD_CANADA_M3':     'MABMM301CAM189S',  # Canada M3 broad money ✓ (OECD)
            'OECD_AUSTRALIA_M3':  'MABMM301AUM189S',  # Australia M3 ✓
            'OECD_KOREA_M2':      'MYAGM2KRM189N',    # South Korea M2 ✓
            'OECD_SWITZERLAND_M3':'MABMM301CHM189S',  # Swiss M3 ✓
            'OECD_SWEDEN_M3':     'MABMM301SEM189S',  # Sweden M3 ✓
        }
        
        df = self.fred.fetch_multiple(oecd_series, Config.START_DATE, Config.END_DATE)
        
        # Calculate growth rates
        for col in df.columns:
            if any(x in col for x in ['M1', 'M2', 'M3', 'M4']):
                df[f"{col}_YoY"] = df[col].pct_change(12) * 100
        
        print(f"✓ Fetched {len([c for c in df.columns if c.startswith('OECD_')])} OECD indicators")
        
        return df
    
    def standardize_frequency(self, df, target_freq='ME'):
        """Standardize all series to target frequency"""
        print(f"\n=== Standardizing to {target_freq} frequency ===")
        
        if df.empty:
            return df
        
        # Resample to target frequency
        if target_freq in ('M', 'ME'):
            df_resampled = df.resample('ME').last()
        elif target_freq in ('W', 'W-FRI', 'W-SUN'):
            df_resampled = df.resample('W').last()
        else:
            df_resampled = df
        
        # Forward fill missing values (up to 3 periods)
        df_resampled = df_resampled.ffill(limit=3)
        
        return df_resampled
    
    def normalize_zscore(self, df, window=36):
        """
        Z-score normalization with rolling window
        window: number of periods for rolling mean/std (default 36 months = 3 years)
        """
        print(f"\n=== Normalizing with Z-score (rolling window={window}) ===")
        
        df_normalized = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            if df[col].notna().sum() > window:
                rolling_mean = df[col].rolling(window=window, min_periods=window//2).mean()
                rolling_std = df[col].rolling(window=window, min_periods=window//2).std()
                df_normalized[col] = (df[col] - rolling_mean) / rolling_std
            else:
                # Not enough data for rolling, use expanding
                expanding_mean = df[col].expanding(min_periods=1).mean()
                expanding_std = df[col].expanding(min_periods=1).std()
                df_normalized[col] = (df[col] - expanding_mean) / (expanding_std + 1e-8)
        
        return df_normalized
    
    # -----------------------------------------------------------------------
    # BLOCK DEFINITIONS
    # Indicators are matched by substring — any column whose name contains
    # one of the listed tokens belongs to that block.
    # n_components: how many PCs to extract for that block.
    # anchor_positive: one representative indicator whose sign should be
    #   positive in a "more liquidity" regime; used to auto-orient PC1.
    # -----------------------------------------------------------------------
    BLOCK_CONFIG = {
        'Monetary_Aggregates': {
            'n_components': 2,
            'anchor_positive': 'US_M2',     # higher M2 → more liquidity
            'members': [
                'US_M2', 'US_CURRENCY',
                'EZ_M2',
                'CN_M2', 'CN_M1', 'CN_M1_M2_RATIO',
                'JP_M2', 'JP_M3', 'JP_REAL_M2',
                'OECD_UK_M4', 'OECD_CANADA_M3', 'OECD_AUSTRALIA_M3',
                'OECD_KOREA_M2', 'OECD_SWITZERLAND_M3', 'OECD_SWEDEN_M3',
            ],
        },
        'CB_Balance_Sheets': {
            'n_components': 1,
            'anchor_positive': 'US_FED_ASSETS',  # larger balance sheet → more liquidity
            'members': [
                'US_FED_ASSETS', 'US_FED_SECURITIES',
                'US_TGA',           # inverted: higher TGA drains liquidity
                'US_RRPONTSYD',     # inverted: higher RRP drains liquidity
                'EZ_ECB_ASSETS',
                'CN_PBOC_ASSETS',
                'JP_BOJ_ASSETS', 'JP_BOJ_GROWTH', 'JP_MONETARY_BASE',
                'GLOBAL_FED_SWAP',
            ],
        },
        'Credit_Spreads': {
            'n_components': 1,
            'anchor_positive': 'GLOBAL_HY_SPREAD_INV',  # tighter spread → more liquidity
            'members': [
                'GLOBAL_HY_SPREAD', 'GLOBAL_HY_SPREAD_INV',
                'GLOBAL_IG_SPREAD', 'GLOBAL_IG_SPREAD_INV',
                'GLOBAL_BBB_SPREAD', 'GLOBAL_BBB_SPREAD_INV',
                'GLOBAL_HY_EU_SPREAD', 'GLOBAL_HY_EU_SPREAD_INV',
                'GLOBAL_TED_SPREAD', 'GLOBAL_TED_SPREAD_INV',
                'GLOBAL_EMV', 'GLOBAL_EMV_INV',
                'GLOBAL_SOFR_INV',
                'US_BANK_CREDIT', 'US_CONSUMER_CREDIT',
                'CN_TOTAL_DEBT', 'CN_DEBT_GROWTH',
            ],
        },
        'FX_Commodities': {
            'n_components': 1,
            'anchor_positive': 'GLOBAL_DXY_INV',   # weaker USD → easier global liquidity
            'members': [
                'GLOBAL_DXY', 'GLOBAL_DXY_INV',
                'GLOBAL_DXY_MAJOR', 'GLOBAL_DXY_MAJOR_INV',
                'GLOBAL_OIL_WTI', 'GLOBAL_OIL_BRENT',
                'GLOBAL_COPPER',
                'CN_EXCHANGE_RATE', 'CN_FX_CHANGE', 'CN_RESERVE_ASSETS',
                'JP_EXCHANGE_RATE', 'JP_FOREX_RESERVES', 'JP_FX_CHANGE',
            ],
        },
        'Macro_Flows': {
            'n_components': 1,
            'anchor_positive': 'CN_EXPORTS',    # stronger trade → higher activity/liquidity demand
            'members': [
                'CN_EXPORTS', 'CN_IMPORTS', 'CN_TRADE_BALANCE', 'CN_GDP',
                'JP_EXPORTS', 'JP_IMPORTS', 'JP_TRADE_BALANCE',
                'JP_INDUSTRIAL_PROD', 'JP_GDP', 'JP_CPI',
                'CN_POLICY_RATE',
                'JP_10Y_YIELD', 'JP_CALL_RATE', 'JP_POLICY_RATE',
                'GLOBAL_FED_FUNDS', 'GLOBAL_3M_TBILL',
            ],
        },
    }

    def _prepare_block(self, block_name: str) -> pd.DataFrame:
        """
        Return the cleaned, normalised sub-DataFrame for one block.
        Columns are selected by exact name match against BLOCK_CONFIG members.
        Only columns that actually exist in self.data are included.
        Requires ≥ 50 % non-NaN coverage; remaining NaNs forward-filled then
        row-dropped so PCA always receives a complete rectangular matrix.
        """
        cfg = self.BLOCK_CONFIG[block_name]
        wanted = set(cfg['members'])
        cols = [c for c in self.data.columns if c in wanted]

        if not cols:
            print(f"  ⚠  {block_name}: no matching columns in data — skipping")
            return pd.DataFrame()

        sub = self.data[cols].copy()

        # Drop columns with < 50 % coverage across the full date range
        min_obs = len(sub) * 0.5
        sub = sub.loc[:, sub.notna().sum() >= min_obs]

        # Forward-fill gaps up to 3 periods so short gaps don't create NaNs
        sub = sub.ffill(limit=3)

        # Drop rows where MORE than half the columns are still missing.
        # (Using dropna() here would discard the entire early history whenever
        # one indicator starts late — that caused the 19-month truncation bug.)
        min_valid = max(2, sub.shape[1] // 2)
        sub = sub.dropna(thresh=min_valid)

        # Impute any remaining NaNs so sklearn PCA receives a fully finite matrix.
        # Strategy (in order):
        #   1. Expanding median  — no look-ahead, works when column has some data
        #   2. Full-column median — fallback when expanding produces NaN (sparse early rows)
        #   3. 0.0               — last resort for columns that are entirely NaN
        for col in sub.columns:
            if not sub[col].isna().any():
                continue
            # Pass 1: expanding median fills from the first non-NaN value onward
            fill = sub[col].expanding(min_periods=1).median()
            sub[col] = sub[col].fillna(fill)
            # Pass 2: if any NaNs remain (column starts with NaN block),
            #         fill with the whole-column median
            if sub[col].isna().any():
                col_median = sub[col].median()
                if pd.isna(col_median):
                    col_median = 0.0          # column is entirely NaN → neutral
                sub[col] = sub[col].fillna(col_median)

        # Drop columns that are still all-NaN (wholly missing series)
        sub = sub.dropna(axis=1, how='all')

        # Hard assertion — surface a clear diagnostic rather than a cryptic sklearn error
        if sub.isna().any().any():
            bad = sub.columns[sub.isna().any()].tolist()
            nan_counts = sub[bad].isna().sum().to_dict()
            print(f"  ⚠  {block_name}: NaNs remain after imputation in: {nan_counts}")
            print(f"     Dropping those columns before PCA.")
            sub = sub.drop(columns=bad)

        if sub.empty or sub.shape[1] < 2:
            print(f"  ⚠  {block_name}: too few valid columns ({sub.shape[1]}) — skipping")
            return pd.DataFrame()

        return sub

    def _orient_components(
        self,
        components: np.ndarray,      # shape (n_components, n_features)
        scores: np.ndarray,           # shape (n_obs, n_components)
        feature_names: list,
        anchor_col: str,
    ) -> np.ndarray:
        """
        Flip PC1 (index 0) so that it has a positive loading on `anchor_col`.
        PC2+ are left as-is (their economic interpretation is secondary).
        Returns the possibly-flipped scores array.
        """
        if anchor_col not in feature_names:
            # Fallback: orient so the mean loading of the first component is positive
            if components[0].mean() < 0:
                scores = scores.copy()
                scores[:, 0] *= -1
            return scores

        anchor_idx = feature_names.index(anchor_col)
        if components[0, anchor_idx] < 0:
            scores = scores.copy()
            scores[:, 0] *= -1
        return scores

    def construct_block_factors(self) -> pd.DataFrame:
        """
        Run PCA independently within each block and return a DataFrame of
        block scores, one column per component.

        Column naming:
          Monetary_Aggregates_PC1, Monetary_Aggregates_PC2,
          CB_Balance_Sheets_PC1,
          Credit_Spreads_PC1,
          FX_Commodities_PC1,
          Macro_Flows_PC1

        PC1 for every block is auto-oriented so that higher = more liquidity
        / risk-on, via the anchor indicator defined in BLOCK_CONFIG.

        Loadings summary is printed to stdout and saved to
        results/block_pca_loadings.csv.
        """
        from sklearn.decomposition import PCA
        import os

        print("\n" + "=" * 60)
        print("BLOCK-PCA FACTOR CONSTRUCTION")
        print("=" * 60)

        if self.data.empty:
            raise RuntimeError("No data — call fetch_all_data() first.")

        all_scores: dict[str, pd.Series] = {}
        all_loadings: list[pd.DataFrame] = []

        for block_name, cfg in self.BLOCK_CONFIG.items():
            n_comp = cfg['n_components']
            anchor = cfg['anchor_positive']
            print(f"\n── {block_name}  (n_components={n_comp}) ──")

            sub = self._prepare_block(block_name)
            if sub.empty:
                continue

            # Clamp n_components to available features
            n_comp = min(n_comp, sub.shape[1])

            pca = PCA(n_components=n_comp)
            raw_scores = pca.fit_transform(sub.values)   # (n_obs, n_comp)
            components = pca.components_                  # (n_comp, n_features)

            # Auto-orient PC1
            raw_scores = self._orient_components(
                components, raw_scores, list(sub.columns), anchor
            )

            # Store each component as a named Series
            for pc_idx in range(n_comp):
                col_name = f"{block_name}_PC{pc_idx + 1}"
                all_scores[col_name] = pd.Series(
                    raw_scores[:, pc_idx], index=sub.index, name=col_name
                )

            # Variance summary
            for pc_idx in range(n_comp):
                ev = pca.explained_variance_ratio_[pc_idx]
                print(f"  PC{pc_idx+1}  explained variance: {ev:.1%}")

            # Loadings table
            for pc_idx in range(n_comp):
                col_name = f"{block_name}_PC{pc_idx + 1}"
                ld = pd.DataFrame({
                    'block':    block_name,
                    'component': col_name,
                    'indicator': sub.columns,
                    'loading':   components[pc_idx],
                    'explained_var': pca.explained_variance_ratio_[pc_idx],
                }).sort_values('loading', ascending=False)
                all_loadings.append(ld)

                top = ld.head(5)['indicator'].tolist()
                bot = ld.tail(3)['indicator'].tolist()
                print(f"  PC{pc_idx+1} top loadings : {top}")
                print(f"  PC{pc_idx+1} low loadings : {bot}")

        # ── Assemble output DataFrame ──────────────────────────────────────
        self.block_scores = pd.concat(all_scores.values(), axis=1).sort_index()

        # Save loadings
        os.makedirs('results', exist_ok=True)
        loadings_df = pd.concat(all_loadings, ignore_index=True)
        loadings_df.to_csv('results/block_pca_loadings.csv', index=False)
        print(f"\n✓ Loadings saved → results/block_pca_loadings.csv")

        # Save scores
        self.block_scores.to_csv('results/block_pca_scores.csv')
        print(f"✓ Scores saved  → results/block_pca_scores.csv")
        print(f"\nBlock scores shape: {self.block_scores.shape}")
        print(f"Columns: {list(self.block_scores.columns)}")

        return self.block_scores

    def plot_block_factors(self) -> None:
        """
        One chart with 5 subplots (one per block), each showing:
          • PC1 as a filled area (green=positive / red=negative)
          • 12-month rolling average as an overlaid line
          • PC2 as a thin dashed line when it exists (Monetary Aggregates)
        Saved to results/block_pca_scores.png
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        import os

        if not hasattr(self, 'block_scores') or self.block_scores.empty:
            print("No block scores — run construct_block_factors() first.")
            return

        # Colour palette per block
        PALETTE = {
            'Monetary_Aggregates':  '#2E86AB',
            'CB_Balance_Sheets':    '#A23B72',
            'Credit_Spreads':       '#F18F01',
            'FX_Commodities':       '#3BB273',
            'Macro_Flows':          '#7B2D8B',
        }
        LABELS = {
            'Monetary_Aggregates': 'Monetary Aggregates',
            'CB_Balance_Sheets':   'CB Balance Sheets',
            'Credit_Spreads':      'Credit Spreads',
            'FX_Commodities':      'FX / Commodities',
            'Macro_Flows':         'Macro Flows',
        }

        block_names = list(self.BLOCK_CONFIG.keys())
        fig, axes = plt.subplots(len(block_names), 1,
                                 figsize=(16, 4 * len(block_names)),
                                 sharex=True)

        for ax, block_name in zip(axes, block_names):
            pc1_col = f"{block_name}_PC1"
            pc2_col = f"{block_name}_PC2"
            colour   = PALETTE[block_name]
            label    = LABELS[block_name]

            if pc1_col not in self.block_scores.columns:
                ax.set_visible(False)
                continue

            s = self.block_scores[pc1_col].dropna()
            roll = s.rolling(12).mean()

            ax.fill_between(s.index, 0, s.values,
                            where=(s.values >= 0),
                            alpha=0.25, color='green')
            ax.fill_between(s.index, 0, s.values,
                            where=(s.values < 0),
                            alpha=0.25, color='red')
            ax.plot(s.index, s.values,
                    color=colour, linewidth=1.2, alpha=0.7)
            ax.plot(roll.index, roll.values,
                    color=colour, linewidth=2.2, label='12M avg')

            # PC2 overlay (Monetary Aggregates only)
            if pc2_col in self.block_scores.columns:
                s2 = self.block_scores[pc2_col].dropna()
                ax.plot(s2.index, s2.values,
                        color='grey', linewidth=1, linestyle='--',
                        alpha=0.6, label='PC2')

            ax.axhline(0, color='black', linewidth=0.7, linestyle='--', alpha=0.4)
            ax.set_ylabel('Score', fontsize=10)
            ax.set_title(label, fontsize=12, fontweight='bold', loc='left')
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8, loc='upper left')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        fig.suptitle('Global Liquidity — Block PCA Scores\n(higher = more liquidity / risk-on)',
                     fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        out = 'results/block_pca_scores.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"✓ Chart saved → {out}")
        plt.close()

    def plot_block_correlation(self) -> None:
        """
        Heatmap of correlations between all block PC scores.
        Useful for checking independence — low cross-block correlation
        validates that the blocks are capturing distinct dimensions.
        Saved to results/block_pca_correlation.png
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os

        if not hasattr(self, 'block_scores') or self.block_scores.empty:
            print("No block scores — run construct_block_factors() first.")
            return

        corr = self.block_scores.corr()
        fig, ax = plt.subplots(figsize=(max(8, len(corr) + 2),
                                        max(6, len(corr) + 1)))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={'shrink': 0.8}, ax=ax,
                    vmin=-1, vmax=1)
        ax.set_title('Block PCA Score Correlations', fontsize=13,
                     fontweight='bold', pad=14)
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        out = 'results/block_pca_correlation.png'
        plt.savefig(out, dpi=150, bbox_inches='tight')
        print(f"✓ Correlation heatmap saved → {out}")
        plt.close()

    def save_factor(self, filename='results/global_liquidity_factor.csv'):
        """Save the legacy single-factor (kept for backward compat)."""
        if hasattr(self, 'factor') and not self.factor.empty:
            import os; os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.factor.to_csv(filename)
            print(f"Factor saved to {filename}")

    def plot_factor(self):
        """Legacy single-factor chart — prefer plot_block_factors()."""
        if not hasattr(self, 'factor') or self.factor.empty:
            print("No legacy factor — use plot_block_factors() instead.")
            return

        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        s = self.factor
        axes[0].plot(s.index, s.values, linewidth=2, color='#2E86AB')
        axes[0].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[0].fill_between(s.index, 0, s.values,
                             where=(s.values > 0), alpha=0.3,
                             color='green', label='Expansionary')
        axes[0].fill_between(s.index, 0, s.values,
                             where=(s.values < 0), alpha=0.3,
                             color='red', label='Contractionary')
        axes[0].set_title('Global Liquidity Factor (Z-Score)',
                          fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Z-Score', fontsize=12)
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        roll = s.rolling(12).mean()
        axes[1].plot(s.index, s.values, alpha=0.4, color='#2E86AB', label='Factor')
        axes[1].plot(roll.index, roll.values, linewidth=2.5, color='#A23B72',
                     label='12M Average')
        axes[1].axhline(0, color='black', linestyle='--', alpha=0.3)
        axes[1].set_title('Global Liquidity Factor — 12M Moving Average',
                          fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Z-Score', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.tight_layout()
        import os; os.makedirs('results', exist_ok=True)
        plt.savefig('results/global_liquidity_factor_chart.png',
                    dpi=300, bbox_inches='tight')
        print("Chart saved to results/global_liquidity_factor_chart.png")
        plt.close()

    def correlation_analysis(self):
        """Heatmap of all raw indicators — prefer plot_block_correlation()."""
        if self.data.empty:
            return
        import matplotlib.pyplot as plt
        import seaborn as sns

        corr_matrix = self.data.corr()
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, square=True, linewidths=0.5,
                    cbar_kws={'shrink': 0.8}, ax=ax)
        ax.set_title('Correlation Matrix of Liquidity Indicators',
                     fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        import os; os.makedirs('results', exist_ok=True)
        plt.savefig('results/liquidity_correlation_matrix.png',
                    dpi=300, bbox_inches='tight')
        print("Correlation matrix saved to results/liquidity_correlation_matrix.png")
        plt.close()

    def fetch_all_data(self):
        """Fetch all liquidity data from all sources"""
        print("=" * 60)
        print("GLOBAL LIQUIDITY FACTOR CONSTRUCTION")
        print("Enhanced China & Japan Coverage + OECD Backup")
        print("=" * 60)
        
        # Fetch from all sources
        us_data = self.fetch_us_liquidity()
        ez_data = self.fetch_eurozone_liquidity()
        cn_data = self.fetch_china_liquidity()
        jp_data = self.fetch_japan_liquidity()
        global_data = self.fetch_global_indicators()
        oecd_data = self.fetch_oecd_liquidity()
        
        # Combine all data
        all_data = pd.concat([us_data, ez_data, cn_data, jp_data, global_data, oecd_data], axis=1)
        
        # Standardize frequency
        all_data = self.standardize_frequency(all_data, Config.TARGET_FREQ)
        
        # Normalize
        all_data_normalized = self.normalize_zscore(all_data)
        
        self.data = all_data_normalized
        
        print(f"\n=== Data Summary ===")
        print(f"Date range: {self.data.index.min()} to {self.data.index.max()}")
        print(f"Number of indicators: {len(self.data.columns)}")
        print(f"Number of observations: {len(self.data)}")
        
        # Regional breakdown
        us_count = len([c for c in self.data.columns if c.startswith('US_')])
        ez_count = len([c for c in self.data.columns if c.startswith('EZ_')])
        cn_count = len([c for c in self.data.columns if c.startswith('CN_')])
        jp_count = len([c for c in self.data.columns if c.startswith('JP_')])
        global_count = len([c for c in self.data.columns if c.startswith('GLOBAL_')])
        oecd_count = len([c for c in self.data.columns if c.startswith('OECD_')])
        
        print(f"\nRegional Breakdown:")
        print(f"  US Indicators:       {us_count:3d}")
        print(f"  Eurozone Indicators: {ez_count:3d}")
        print(f"  China Indicators:    {cn_count:3d}")
        print(f"  Japan Indicators:    {jp_count:3d}")
        print(f"  Global Indicators:   {global_count:3d}")
        print(f"  OECD Indicators:     {oecd_count:3d}")
        print(f"  {'─' * 30}")
        print(f"  Total:               {len(self.data.columns):3d}")
        
        return self.data
    
    def save_data(self, filename='results/global_liquidity_data.csv'):
        """Save the raw and normalized data"""
        if not self.data.empty:
            import os; os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.data.to_csv(filename)
            print(f"\nData saved to {filename}")
    
    def save_factor(self, filename='results/global_liquidity_factor.csv'):
        """Save the factor"""
        if not self.factor.empty:
            import os; os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.factor.to_csv(filename)
            print(f"Factor saved to {filename}")
    
    def plot_factor(self):
        """Create visualization of the liquidity factor"""
        if self.factor.empty:
            print("Factor not yet constructed.")
            return
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # Plot 1: Liquidity Factor
        axes[0].plot(self.factor.index, self.factor.values, linewidth=2, color='#2E86AB')
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[0].fill_between(self.factor.index, 0, self.factor.values, 
                            where=(self.factor.values > 0), alpha=0.3, color='green', label='Expansionary')
        axes[0].fill_between(self.factor.index, 0, self.factor.values, 
                            where=(self.factor.values < 0), alpha=0.3, color='red', label='Contractionary')
        axes[0].set_title('Global Liquidity Factor (Z-Score)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Z-Score', fontsize=12)
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        # Plot 2: Rolling 12-month average
        rolling_avg = self.factor.rolling(window=12).mean()
        axes[1].plot(self.factor.index, self.factor.values, alpha=0.4, color='#2E86AB', label='Factor')
        axes[1].plot(rolling_avg.index, rolling_avg.values, linewidth=2.5, color='#A23B72', label='12M Average')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1].set_title('Global Liquidity Factor with 12-Month Moving Average', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Z-Score', fontsize=12)
        axes[1].set_xlabel('Date', fontsize=12)
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        
        plt.tight_layout()
        import os; os.makedirs('results', exist_ok=True)
        plt.savefig('results/global_liquidity_factor_chart.png', dpi=300, bbox_inches='tight')
        print("\nChart saved to results/global_liquidity_factor_chart.png")
        plt.close()
    
    def correlation_analysis(self):
        """Analyze correlations between indicators"""
        if self.data.empty:
            return
        
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Calculate correlation matrix
        corr_matrix = self.data.corr()
        
        # Plot heatmap
        fig, ax = plt.subplots(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix of Liquidity Indicators', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        import os; os.makedirs('results', exist_ok=True)
        plt.savefig('results/liquidity_correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("Correlation matrix saved to results/liquidity_correlation_matrix.png")
        plt.close()


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

def main():
    """Main execution — block-PCA pipeline."""

    fred_api_key = Config.FRED_API_KEY
    if fred_api_key == "YOUR_FRED_API_KEY_HERE":
        print("\n" + "=" * 60)
        print("⚠️  Set your FRED API key in Config.FRED_API_KEY")
        print("   https://fred.stlouisfed.org/docs/api/api_key.html")
        print("=" * 60 + "\n")
        return

    glf = GlobalLiquidityFactor(fred_api_key)

    # 1. Fetch & normalise all raw indicators
    data = glf.fetch_all_data()
    if data.empty:
        print("No data fetched — check API key and network.")
        return

    # 2. Save raw normalised indicators
    glf.save_data('results/global_liquidity_data.csv')

    # 3. Run block-PCA → 6 scores (5 blocks, Monetary gets PC1 + PC2)
    block_scores = glf.construct_block_factors()

    # 4. Charts
    glf.plot_block_factors()
    glf.plot_block_correlation()

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("✓  BLOCK-PCA CONSTRUCTION COMPLETE")
    print("=" * 60)
    print("\nOutputs in results/")
    print("  global_liquidity_data.csv    — all raw normalised indicators")
    print("  block_pca_scores.csv         — 6 block PC scores (time series)")
    print("  block_pca_loadings.csv       — loadings + explained variance")
    print("  block_pca_scores.png         — 5-panel factor chart")
    print("  block_pca_correlation.png    — inter-block correlation heatmap")

    print("\nLatest block scores:")
    latest = block_scores.iloc[-1]
    for col, val in latest.items():
        bar = '▲' if val > 0 else '▼'
        print(f"  {bar}  {col:<35s}  {val:+.3f}")

    print(f"\nDate: {block_scores.index[-1].strftime('%Y-%m-%d')}")


if __name__ == "__main__":
    main()
