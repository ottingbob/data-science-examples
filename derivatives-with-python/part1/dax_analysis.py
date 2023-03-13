# import pandas.io.data as web
import numpy as np
import yfinance as yf
from gbm_analysis import *


# read historical DAX data from yahoo finance, calculate log returns,
# realized variance and volatility
def read_dax_data():
    DAX = yf.download("^GDAXI", start="2004-09-30", end="2014-09-30")
    print(DAX)
    # DAX = web.DataReader(
    # "^GDAXI", data_source="yahoo", start="30-09-2004", end="30-09-2014"
    # )
    DAX.rename(columns={"Adj Close": "index"}, inplace=True)
    DAX["returns"] = np.log(DAX["index"] / DAX["index"].shift(1))
    DAX["rea_var"] = 252 * np.cumsum(DAX["returns"] ** 2) / np.arange(len(DAX))
    DAX["rea_vol"] = np.sqrt(DAX["rea_var"])
    DAX = DAX.dropna()
    return DAX


# Count number of return jumps as defined in size by value.
def count_jumps(data, value):
    jumps = np.sum(np.abs(data["returns"]) > value)
    return jumps


if __name__ == "__main__":
    dax = read_dax_data()
    print(dax)
    print_statistics(dax)
    quotes_returns(dax)
    return_histogram(dax)
    return_qq_chart(dax)
    realized_volatility(dax)
    rolling_statistics(dax)
    plt.show()
