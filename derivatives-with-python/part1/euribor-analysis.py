# Analyze Euribor interest rate data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_euribor_data() -> pd.DataFrame:
    # Read historical Euribor data from excel file and calculate log returns
    EBO = pd.read_excel(
        "./derivatives-with-python/part1/EURIBOR_current.xlsx",
        index_col=0,
        parse_dates=True,
    )
    EBO["returns"] = np.log(EBO["1w"] / EBO["1w"].shift(1))
    EBO = EBO.dropna()
    return EBO


def plot_term_structure(data: pd.DataFrame):
    markers = [".", "-.", "-", "o"]
    for i, mat in enumerate(["1w", "1m", "6m", "12m"]):
        plt.plot(data[mat].index, data[mat].values, f"b{markers[i]}", label=mat)

    plt.grid()
    plt.legend()
    plt.xlabel("strike")
    plt.ylabel("implied volatility")
    plt.ylim(0.0, plt.ylim()[1])
    plt.show()


if __name__ == "__main__":
    data = read_euribor_data()
    plot_term_structure(data)
