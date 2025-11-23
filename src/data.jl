using CSV, DataFrames, Dates, TimeZones

function get_pair_interval_df(pair,interval)
    #loads kraken CSV and returns formated df
    colnames = [:timestamp, :open, :high, :low, :close, :volume, :trades]
    file=pwd()*"//src//data//Kraken_OHLCVT//"*pair*"_"*interval*".csv"
    if has_header(file)#if headers
        df = CSV.read(file, DataFrame)
    else#if no headers
        df = CSV.read(file, DataFrame, header=false)
    end
    return rename!(df, colnames)
end

#check if the file being loadded has headers or not
function has_header(file)
    firstline = open(file) do io
        readline(io)
    end

    # Split by comma
    fields = split(firstline, ',')

    # Try parsing numbers; headers will fail
    all(field -> try
            parse(Float64, field); true
        catch
            false
        end, fields) ? false : true
end

#Open - the first traded price
#High - the highest traded price
#Low - the lowest traded price
#Close - the final traded price
#Volume - the total volume traded by all trades
#Trades - the number of individual trades
#1, 5, 15, 30, 60, 240, 720 and 1440 minute intervals
#Data spans: 2015-08-07T14:03:00+02:00 - 2025-03-31T23:59:00+02:00

pair="ETHUSD"
interval="5"

ETH_df=get_pair_interval_df(pair,interval)
    


ts = ETH_df[!,:timestamp][1]
dt = unix2datetime(ts)
dt_be = ZonedDateTime(dt, tz"Europe/Brussels")
println(dt_be)

pair="BTCUSD"
interval="5"

BTC_df=get_pair_interval_df(pair,interval)
ts = BTC_df[!,:timestamp][1]
dt = unix2datetime(ts)
dt_be = ZonedDateTime(dt, tz"Europe/Brussels")
println(dt_be)
    
