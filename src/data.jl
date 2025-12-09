function get_data_path_from_script()
    # If the script is in /ProjectRoot/tests/, we go up one level (..), 
    # then into src/data/...
    # If the script is in /ProjectRoot/src/, we simply go into data/...
    
    # We will assume the data is 3 levels up from the script location 
    # (e.g., if code is in tests/run_tests.jl)
    
    # A cleaner approach is to find the project root from the script's location.
    
    # Simple relative path based on current directory of THIS script:
    # Assuming this script is in /ProjectRoot/path_fixer.jl
    # To get to /ProjectRoot/src/data/Kraken_OHLCVT/
    return joinpath(@__DIR__, "src", "data", "Kraken_OHLCVT", FILENAME)
end

function get_pair_interval_df(pair,interval)
    #loads kraken CSV and returns formated df
    colnames = [:timestamp, :open, :high, :low, :close, :volume, :trades]

    FILENAME = string(pair, "_", interval, ".csv")
    file=joinpath(@__DIR__, "data", "Kraken_OHLCVT", FILENAME)
    
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
