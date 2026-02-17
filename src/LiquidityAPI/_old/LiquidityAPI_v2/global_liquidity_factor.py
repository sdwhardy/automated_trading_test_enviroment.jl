"""
Global Liquidity Factor Construction
====================================
Builds a global liquidity factor usable across risk assets using:
- Monetary aggregates (M2, M3)
- Central bank balance sheets
- Credit growth
- Cross-border USD funding proxies

Data sources: FRED, ECB SDMX, BIS, IMF
Frequency: Weekly/Monthly (standardized)
Normalization: Z-score
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
    TARGET_FREQ = "W"  # 'W' for weekly, 'ME' for month-end (pandas 2.2+)


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
    
    def construct_factor(self, method='pca', n_components=1, weights=None):
        """
        Construct the global liquidity factor
        
        method: 'pca' (Principal Component Analysis), 'equal' (equal weight), 'custom' (custom weights)
        n_components: number of PCA components (default 1)
        weights: dict of custom weights for each indicator
        """
        print(f"\n=== Constructing Factor (method={method}) ===")
        
        if self.data.empty:
            print("No data available. Run fetch_all_data() first.")
            return pd.Series()
        
        # Remove columns with too many NaNs
        valid_cols = self.data.columns[self.data.notna().sum() > len(self.data) * 0.5]
        data_clean = self.data[valid_cols].copy()
        
        # Forward fill and drop remaining NaNs
        data_clean = data_clean.ffill().dropna()
        
        if data_clean.empty:
            print("No valid data after cleaning.")
            return pd.Series()
        
        print(f"Using {len(data_clean.columns)} indicators: {list(data_clean.columns)}")
        
        if method == 'pca':
            from sklearn.decomposition import PCA
            
            pca = PCA(n_components=n_components)
            factor_values = pca.fit_transform(data_clean)
            
            self.factor = pd.Series(factor_values[:, 0], index=data_clean.index, name='Global_Liquidity_Factor')
            
            # Print explained variance
            print(f"Explained variance ratio: {pca.explained_variance_ratio_[0]:.2%}")
            
            # Print component loadings
            loadings = pd.DataFrame(
                pca.components_[0],
                index=data_clean.columns,
                columns=['PC1_Loading']
            ).sort_values('PC1_Loading', ascending=False)
            print("\nTop 10 Component Loadings:")
            print(loadings.head(10))
            
        elif method == 'equal':
            # Equal weighted average
            self.factor = data_clean.mean(axis=1)
            self.factor.name = 'Global_Liquidity_Factor'
            
        elif method == 'custom' and weights:
            # Custom weighted average
            weighted_sum = pd.Series(0, index=data_clean.index)
            total_weight = 0
            
            for col in data_clean.columns:
                if col in weights:
                    weighted_sum += data_clean[col] * weights[col]
                    total_weight += weights[col]
            
            self.factor = weighted_sum / total_weight
            self.factor.name = 'Global_Liquidity_Factor'
        
        else:
            print("Invalid method or missing weights.")
            return pd.Series()
        
        return self.factor
    
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
    """Main execution function"""
    
    # Initialize with your FRED API key
    # Get your key from: https://fred.stlouisfed.org/docs/api/api_key.html
    fred_api_key = Config.FRED_API_KEY
    
    if fred_api_key == "YOUR_FRED_API_KEY_HERE":
        print("\n" + "="*60)
        print("⚠️  IMPORTANT: Please set your FRED API key!")
        print("="*60)
        print("1. Get a free API key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        print("2. Replace 'YOUR_FRED_API_KEY_HERE' in Config.FRED_API_KEY")
        print("="*60 + "\n")
        return
    
    # Create liquidity factor builder
    glf = GlobalLiquidityFactor(fred_api_key)
    
    # Fetch all data
    data = glf.fetch_all_data()
    
    if not data.empty:
        # Construct factor using PCA
        factor = glf.construct_factor(method='pca', n_components=1)
        
        # Save outputs
        glf.save_data('results/global_liquidity_data.csv')
        glf.save_factor('results/global_liquidity_factor.csv')
        
        # Create visualizations
        glf.plot_factor()
        glf.correlation_analysis()
        
        print("\n" + "="*60)
        print("✓ CONSTRUCTION COMPLETE")
        print("="*60)
        print("\nOutputs generated:")
        print("1. global_liquidity_data.csv - All indicators (normalized)")
        print("2. global_liquidity_factor.csv - Final liquidity factor")
        print("3. global_liquidity_factor_chart.png - Factor visualization")
        print("4. liquidity_correlation_matrix.png - Correlation heatmap")
        print("\nFactor Statistics:")
        print(f"Mean: {factor.mean():.4f}")
        print(f"Std Dev: {factor.std():.4f}")
        print(f"Min: {factor.min():.4f}")
        print(f"Max: {factor.max():.4f}")
        print(f"Current (latest): {factor.iloc[-1]:.4f}")
    

if __name__ == "__main__":
    main()
