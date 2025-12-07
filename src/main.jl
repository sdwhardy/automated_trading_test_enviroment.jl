using CSV, DataFrames, Dates, TimeZones, Statistics
include("clusteringIndicators.jl")
include("data.jl")


#Open - the first traded price
#High - the highest traded price
#Low - the lowest traded price
#Close - the final traded price
#Volume - the total volume traded by all trades
#Trades - the number of individual trades
#1, 5, 15, 30, 60, 240, 720 (12 hour) and 1440 (24 hour) minute intervals
#BTC Data spans: 2013-10-06T21:34:00+02:00 - 2025-03-31T23:59:00+02:00
#ETH Data spans: 2015-08-07T14:03:00+02:00 - 2025-03-31T23:59:00+02:00


OHLCVT=Dict()
pair="ETHUSD"
interval="1440"
push!(OHLCVT,pair=>Dict(interval=>Dict("df"=>get_pair_interval_df(pair,interval),"dict"=>Dict())))
#interval="720"
#push!(OHLCVT[pair],interval=>Dict("df"=>get_pair_interval_df(pair,interval),"dict"=>Dict()))

#=ts = OHLCVT["ETHUSD"]["1440"]["df"][!,:timestamp][1]
dt = unix2datetime(ts)
dt_be = ZonedDateTime(dt, tz"Europe/Brussels")
println(dt_be)=#

OHLCVT["ETHUSD"]["1440"]["df"]=calculate_clustering_indicators(OHLCVT["ETHUSD"]["1440"]["df"])

#=NOTE: next step Validate!!!!!!!!!!!!
validation:
1) random calc of raw values 
ln return (1 day and 21 day)
69:OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn21day][69]==log(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][69]/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][48])
149:OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn1day][149]==log(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][149]/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][148])
1512:OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn21day][1512]==log(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][1512]/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][1491])
2289:OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn1day][2289]==log(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][2289]/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][2288])
3519:OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn21day][3519]==log(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][3519]/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][3498])

roc (21 day and 63 day)
69:OHLCVT["ETHUSD"]["1440"]["df"][!,:roc21day][69]==(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][69]-OHLCVT["ETHUSD"]["1440"]["df"][!,:close][48])/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][48]
149:OHLCVT["ETHUSD"]["1440"]["df"][!,:roc63day][149]==(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][149]-OHLCVT["ETHUSD"]["1440"]["df"][!,:close][86])/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][86]
1512:OHLCVT["ETHUSD"]["1440"]["df"][!,:roc21day][1512]==(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][1512]-OHLCVT["ETHUSD"]["1440"]["df"][!,:close][1491])/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][1491]
2289:OHLCVT["ETHUSD"]["1440"]["df"][!,:roc63day][2289]==(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][2289]-OHLCVT["ETHUSD"]["1440"]["df"][!,:close][2226])/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][2226]
3519:OHLCVT["ETHUSD"]["1440"]["df"][!,:roc21day][3519]==(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][3519]-OHLCVT["ETHUSD"]["1440"]["df"][!,:close][3498])/OHLCVT["ETHUSD"]["1440"]["df"][!,:close][3498]


3) Realized Volatility
​5, 10, 21 day windows
t=69;n=10
69:OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol10day][t]==sqrt(sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn1day][t-n+1:t].^2)/n*252)
t=149; n=5
149:OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol5day][t]==sqrt(sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn1day][t-n+1:t].^2)/n*252)
t=1512; n=10
1512:OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol10day][t]==sqrt(sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn1day][t-n+1:t].^2)/n*252)
t=2289;n=21
2289:OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol21day][t]==sqrt(sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn1day][t-n+1:t].^2)/n*252)
t=3519
3519:OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol21day][t]==sqrt(sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn1day][t-n+1:t].^2)/n*252)

4) Garman–Klass volatility (21-day)
gkt=[]
T = 3519
n = 21

for t in T-n:1:T-1
    # Correct Assignment
    high_price  = OHLCVT["ETHUSD"]["1440"]["df"][!,:high][t] # Assign to a new variable name
    low_price   = OHLCVT["ETHUSD"]["1440"]["df"][!,:low][t]
    close_price = OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t]
    open_price  = OHLCVT["ETHUSD"]["1440"]["df"][!,:open][t]

    # Use the correct variable names in the formula
    log_hl = log(high_price / low_price)
    log_co = log(close_price / open_price)
    const_factor = 2*log(2) - 1

    # Calculation: Variance contribution for one day
    variance_contribution = 0.5 * log_hl^2 - const_factor * log_co^2

    push!(gkt, variance_contribution)
end

# Final Garman-Klass Variance (per period)
final_gk_variance = sqrt((sum(gkt) / (n))*252)#1.0623067956701226
OHLCVT["ETHUSD"]["1440"]["df"][!,:gkVol21day][T]#1.0623067956701226

Vol of Vol
t = 69
n = 1
vol=realized_volatility(OHLCVT["ETHUSD"]["1440"]["df"][!,:close],1)
vov=realized_volatility(vol,21)
OHLCVT["ETHUSD"]["1440"]["df"][!,:close][1:23]
OHLCVT["ETHUSD"]["1440"]["df"][!,:volOfVol21day][23:30]
sqrt(sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:lnReturn1day].^2)/n)
23	22.9056
24	22.7531
25	22.7986
26	23.0960
27	22.0080
28	20.3337
29	15.8156
30	16.7019

difference of moving average
10 - 50
t=653
OHLCVT["ETHUSD"]["1440"]["df"][!,:maDiff10_50day][t]
sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-10+1:t])/10-sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-50+1:t])/50

100 - 200
t=2267
OHLCVT["ETHUSD"]["1440"]["df"][!,:maDiff20_100day][t]
sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-20+1:t])/20-sum(OHLCVT["ETHUSD"]["1440"]["df"][!,:close][t-100+1:t])/100

t=69;T=21
OHLCVT["ETHUSD"]["1440"]["df"][!,:volumeZscore21day][t]
vt=OHLCVT["ETHUSD"]["1440"]["df"][!,:volume][t]
vm=mean(OHLCVT["ETHUSD"]["1440"]["df"][!,:volume][t-T:t-1])
vs=std(OHLCVT["ETHUSD"]["1440"]["df"][!,:volume][t-T:t-1])
zscore=(vt-vm)/vs
=#

#=
Transform 
Amihud illiquidity:
→ into log(1 + amihud)

Volume:
You already have a Z-score, so no need to standardize again.

Volatility and vol-of-vol:
Often stabilized with
→ log(vol)
→ then z-score


Apply z-score standardization to all remaining raw features
=#

OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol5day]=log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol5day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol10day]=log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol10day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol21day]=log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:realVol21day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:gkVol21day]=log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:gkVol21day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:volOfVol21day]=log.(OHLCVT["ETHUSD"]["1440"]["df"][!,:volOfVol21day])
OHLCVT["ETHUSD"]["1440"]["df"][!,:amihud21day]=log.(1.0 .+ OHLCVT["ETHUSD"]["1440"]["df"][!,:amihud21day])

exclude = [:timestamp, :open, :high, :low, :close, :volume, :trades, :volumeZscore21day]
cols = Symbol.(names(OHLCVT["ETHUSD"]["1440"]["df"], Not(exclude)))
OHLCVT["ETHUSD"]["1440"]["df"],OHLCVT["ETHUSD"]["1440"]["dict"]=zscore_df(OHLCVT["ETHUSD"]["1440"]["df"],cols)


#################################################################################################################

pair="XBTUSD"
interval="1440"
push!(OHLCVT,pair=>Dict(interval=>Dict("df"=>get_pair_interval_df(pair,interval),"dict"=>Dict())))
#interval="720"
#push!(OHLCVT[pair],interval=>Dict("df"=>get_pair_interval_df(pair,interval),"dict"=>Dict()))


OHLCVT["XBTUSD"]["1440"]["df"]=calculate_clustering_indicators(OHLCVT["XBTUSD"]["1440"]["df"])

#=validation:
1) random calc of raw values 

69
149
1512
2289
3582
=#

#=
Transform 
Amihud illiquidity:
→ into log(1 + amihud)

Volume:
You already have a Z-score, so no need to standardize again.

Volatility and vol-of-vol:
Often stabilized with
→ log(vol)
→ then z-score


Apply z-score standardization to all remaining raw features
=#
OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol5day]=log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol5day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol10day]=log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol10day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol21day]=log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:realVol21day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:gkVol21day]=log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:gkVol21day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:volOfVol21day]=log.(OHLCVT["XBTUSD"]["1440"]["df"][!,:volOfVol21day])
OHLCVT["XBTUSD"]["1440"]["df"][!,:amihud21day]=log.(1.0 .+ OHLCVT["XBTUSD"]["1440"]["df"][!,:amihud21day])

exclude = [:timestamp, :open, :high, :low, :close, :volume, :trades, :volumeZscore21day]
cols = Symbol.(names(OHLCVT["XBTUSD"]["1440"]["df"], Not(exclude)))
OHLCVT["XBTUSD"]["1440"]["df"],OHLCVT["XBTUSD"]["1440"]["dict"]=zscore_df(OHLCVT["XBTUSD"]["1440"]["df"],cols)

#=validation
The mean should be within [-0.05, +0.05]
The std should be within [0.98, 1.02]
=#
