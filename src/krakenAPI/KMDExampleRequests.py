# Endpoints that do not require authentication,
from datetime import datetime, timezone
import KrakenMarketData #support functions



# -----------------------------------------------------------
# SYSTEM INFORMATION (Kraken Public API)
# -----------------------------------------------------------
# Get current system status (online / maintenance / degraded).
# Docs: https://docs.kraken.com/rest/#operation/getSystemStatus
#print(KrakenMarketData.GetSystemStatus().read().decode())

# Get Kraken server time (UNIX + RFC3339). Useful to sync clocks.
# Docs: https://docs.kraken.com/rest/#operation/getServerTime
#print(KrakenMarketData.GetServerTime().read().decode())


# -----------------------------------------------------------
# ASSET INFORMATION
# -----------------------------------------------------------
# Query information about assets (e.g., BTC, ETH).
# "aclass": "currency" limits to normal cryptocurrencies.
# Docs: https://docs.kraken.com/rest/#operation/getAssets
query = {
    #"asset": "BTC",
    #"aclass": "currency"    # asset class = regular cryptocurrency
}
#assets=KrakenMarketData.GetAssetInfo(query).read().decode()
#print(assets)

# -----------------------------------------------------------
# TRADABLE ASSET PAIRS
# -----------------------------------------------------------
# Get info on trading pairs available in a specific region.
# "country_code": "BE" filters to assets available in Belgium.
# "info": which details to return (fees, leverage, margin, etc.)
# Docs: https://docs.kraken.com/rest/#operation/getTradableAssetPairs
query = {
    "pair": "BTC/USD",
    "aclass_base": "currency",   # "currency" or "tokenized_asset"
    "info": "info",              # return basic pair info
    "country_code": "BE"         # restrict to Belgium
}
print(KrakenMarketData.GetTradableAssetPairs(query).read().decode())


# -----------------------------------------------------------
# TICKER INFORMATION
# -----------------------------------------------------------
# Ticker = last trade, bid/ask, volume, VWAP, etc.
# Docs: https://docs.kraken.com/rest/#operation/getTickerInformation
query = {
    "pair": "BTC/EUR",
    "asset_class": "forex"   # "forex" for currency pairs, or "tokenized_asset"
}
btc_eur=KrakenMarketData.GetTickerInformation(query).read().decode()

query = {
    "pair": "ETH/EUR",
    "asset_class": "forex"   # "forex" for currency pairs, or "tokenized_asset"
}
eth_eur=KrakenMarketData.GetTickerInformation(query).read().decode()

query = {
    "pair": "XRP/EUR",
    "asset_class": "forex"   # "forex" for currency pairs, or "tokenized_asset"
}
xrp_eur=KrakenMarketData.GetTickerInformation(query).read().decode()


# -----------------------------------------------------------
# TIMESTAMP HANDLING (UTC → UNIX seconds)
# -----------------------------------------------------------
# Create a timezone-aware datetime in UTC.
dt = datetime(2015, 11, 16, 52, 0, 0, tzinfo=timezone.utc)

# Convert to UNIX timestamp (Kraken expects seconds since epoch).
unix_ts = dt.replace(tzinfo=timezone.utc).timestamp()

# Convert UNIX timestamp back to UTC datetime (verification step).
dt_utc = datetime.fromtimestamp(unix_ts, timezone.utc)
print(dt_utc)


# -----------------------------------------------------------
# OHLC CANDLE DATA
# -----------------------------------------------------------
# "interval": candlestick size in minutes.
# "since": return data AFTER the given UNIX timestamp.
# Docs: https://docs.kraken.com/rest/#operation/getOHLC
query = {
    "pair": "BTC/USD",
    "interval": 1,          # 1-minute candles
    "since": unix_ts,       # return candles after this time
    "asset_class": "forex"  # forex = fiat currency pair
}
KrakenMarketData.GetOHLCData(query).read().decode()


# -----------------------------------------------------------
# ORDER BOOK (MARKET DEPTH)
# -----------------------------------------------------------
# "count": number of levels to return on each side.
# Docs: https://docs.kraken.com/rest/#operation/getOrderBook
query = {
    "pair": "BTC/USD",
    "count": 1,              # top 1 bid + ask level
    "asset_class": "forex"
}
print(KrakenMarketData.GetOrderBook(query).read().decode())


# -----------------------------------------------------------
# RECENT TRADES
# -----------------------------------------------------------
# Returns executed trades after "since".
# Docs: https://docs.kraken.com/rest/#operation/getRecentTrades
dt = datetime(2025, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
unix_ts = dt.replace(tzinfo=timezone.utc).timestamp()

query = {
    "pair": "BTC/USD",
    "since": unix_ts,       # return only trades after this timestamp
    "count": 1,             # limit trades returned
    "asset_class": "forex"
}
print(KrakenMarketData.GetRecentTrades(query).read().decode())


# -----------------------------------------------------------
# RECENT SPREADS (Bid–Ask spread over time)
# -----------------------------------------------------------
# Very useful for backtesting realistic slippage.
# Docs: https://docs.kraken.com/rest/#operation/getRecentSpreads
dt = datetime(2025, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
unix_ts = dt.replace(tzinfo=timezone.utc).timestamp()

query = {
    "pair": "BTC/USD",
    "since": unix_ts,        # return spreads after timestamp
    "asset_class": "forex"
}
print(KrakenMarketData.GetRecentSpreads(query).read().decode())


print("done")