

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
