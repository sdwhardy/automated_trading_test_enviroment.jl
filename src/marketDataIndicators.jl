using MarketData
using TimeSeries
using Indicators


# Load built-in financial data (e.g., Apple stock prices)
#seems to pull totally up to date and almost any ticker...
data = yahoo(:TSLA)

#moving averages
#=
SMA (simple moving average)
WMA (weighted moving average)
EMA (exponential moving average)
TRIMA (triangular moving average)
KAMA (Kaufman adaptive moving average)
MAMA (MESA adaptive moving average, developed by John Ehlers)
HMA (Hull moving average)
ALMA (Arnaud-Legoux moving average)
SWMA (sine-weighted moving average)
DEMA (double exponential moving average)
TEMA (triple exponential moving average)
ZLEMA (zero-lag exponential moving average)
MMA (modified moving average)
VWMA (volume-weighted moving average)
MLR (moving linear regression)
-   Prediction
-   Slope
-   Intercept
-   Standard error
-   Upper & lower bound
-   R-squared
=#
_sma = Indicators.sma(values(data[:Close]), n=3)#simple moving average
_wma = Indicators.wma(values(data[:Close]), n=3)#weighted moving average
_ema = Indicators.ema(values(data[:Close]), n=3)#exponential moving average
_trima = Indicators.trima(values(data[:Close]), n=3)#triangular moving average
_kama = Indicators.kama(values(data[:Close]), n=3)#kaufman adaptive moving average
_mama = Indicators.mama(values(data[:Close]))#MESA adaptive moving average
_hma = Indicators.hma(values(data[:Close]))#HMA (Hull moving average)




#Momentum Indicators
#=
Momentum (n-day price change)
ROC (rate of change)
MACD (moving average convergence-divergence)
RSI (relative strength index)
ADX (average directional index)
Heikin-Ashi (Heiken Ashi)
Parabolic SAR (stop and reverse)
Fast & slow stochastics
SMI (stochastic momentum indicator)
KST (Know Sure Thing)
Williams %R
CCI (commodity channel index)
Donchian channel
Ichimoku Kinko Hyo
Aroon indicator + oscillator
=#
_momentum = Indicators.momentum(values(data[:Close]))#Momentum (n-day price change)
_roc = Indicators.roc(values(data[:Close]))#ROC (rate of change)
_macd = Indicators.macd(values(data[:Close]))#macd (rate of change)


