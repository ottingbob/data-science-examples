import operator
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
from scipy.interpolate import make_interp_spline

sns.set()


class TermColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    # FAIL = "\033[91m"
    FAIL = "\033[31m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @classmethod
    def _log_failure(cls, e: Exception) -> None:
        print(f"{cls.FAIL}{cls.BOLD}{e}{cls.ENDC}")

    @classmethod
    @contextmanager
    def with_failures(cls):
        try:
            yield
        except Exception as e:
            cls._log_failure(e)


# First start by getting data into a dataframe
# course_challenge_file = Path(__name__)
course_challenge_file = Path(
    str(Path(__file__).parent) + "/93.+Course-challenge-before-solutions.xlsx"
)

rev_col = "Revenue ($ 000')"
cogs_col = "Cogs ($ 000')"
pd_df = pd.read_excel(
    course_challenge_file,
    sheet_name="Data",
    header=3,
    usecols=range(1, 6),
    names=[
        "Period",
        "Type of client",
        "Client name",
        rev_col,
        cogs_col,
    ],
)
df = pl.from_pandas(pd_df)

# Task 1:
# Calculate the companies annual revenues and cogs, then provide the companies gross profit
# which is equal to revenues minus cogs

# First double check the dates we are dealing with are only in a single year
assert df.select(["Period"]).unique().shape == (12, 1)

annual_sums = df.with_columns(pl.all()).sum()
annual_revenue = annual_sums.select(pl.col(rev_col)).item()
annual_cogs = annual_sums.select(pl.col(cogs_col)).item()
annual_gross_profit = annual_revenue - annual_cogs
print(f"Annual Revenue 2015: {annual_revenue:.2f}")
print(f"Annual Cogs 2015: {annual_cogs:.2f}")
print(f"Annual Gross Profit 2015: {annual_gross_profit:.2f}")

# Task 2:
# Please provide a breakdown that shows monthly Revenues and Cogs and calculate monthly Gross Profit.
# Create an area chart that shows the development of Revenues and Cogs.
monthly_rev_cogs = (
    df.with_columns(pl.col("Period"))
    .groupby_dynamic("Period", every="1mo", period="1mo", closed="left")
    .agg(
        [
            pl.col(rev_col).sum().alias("Monthly Revenue"),
            pl.col(cogs_col).sum().alias("Monthly Cogs"),
        ]
    )
    .with_columns(
        pl.struct([pl.col("Monthly Revenue"), pl.col("Monthly Cogs")])
        .apply(lambda x: x["Monthly Revenue"] - x["Monthly Cogs"])
        .alias("Monthly Gross Profit")
    )
)


def plot_monthly_rev_cogs(smooth_gross_profit: bool = False):
    fig = plt.figure(figsize=(8, 5))
    # in order to update the ticks we need to add the subplot
    ax = fig.add_subplot(111)
    plt.title("Revenues, Cogs, and Gross Profit FY15")
    plt.xlabel("Time Period", fontsize=12)
    plt.ylabel("Revenue / Cogs", fontsize=12)
    fig.canvas.manager.set_window_title("Task 2: Area chart -- Revenues & Cogs")

    month_labels = (
        monthly_rev_cogs.select(pl.col("Period").cast(pl.Date).dt.strftime("%m/%d/%Y"))
        .get_columns()[0]
        .to_list()
    )

    plt.yticks(fontsize=8)
    ax.set_xticklabels(month_labels, rotation=40, ha="right", fontsize=8)

    cmap = sns.color_palette("mako", 8)
    plt.fill_between(
        month_labels, monthly_rev_cogs["Monthly Revenue"], alpha=0.5, color=cmap[5]
    )
    plt.fill_between(
        month_labels, monthly_rev_cogs["Monthly Cogs"], alpha=0.6, color=cmap[2]
    )
    # Attempt to smooth line
    if smooth_gross_profit:
        spl = make_interp_spline(
            range(len(month_labels)), monthly_rev_cogs["Monthly Gross Profit"], k=3
        )
        smooth = spl(np.linspace(0, 1, 12))
        plt.plot(month_labels, smooth, lw=2, color=cmap[0])
    else:
        plt.plot(
            month_labels, monthly_rev_cogs["Monthly Gross Profit"], lw=2, color=cmap[0]
        )

    plt.legend(["Revenue", "Cogs", "Gross Profit"], loc="lower left")
    plt.tight_layout()
    plt.show()


# plot_monthly_rev_cogs(smooth_gross_profit=False)

# Task 3:
# Please provide a monthly breakdown of Revenues by type of client and calculate the percentage
# incidence that each client type has on the company's Revenues.
# Create a chart that shows the incidence on Revenues that different type of clients had throughout the year.
monthly_rev_by_client = (
    df.with_columns(pl.col("Period"))
    .groupby("Period", "Client name")
    .agg(
        [
            pl.col(rev_col).sum().alias("Monthly Revenue"),
        ]
    )
    .with_columns(
        pl.col("Period").cast(pl.Date).dt.strftime("%m/%d/%Y"),
        pl.sum("Monthly Revenue").over(["Period"]).alias("Total Monthly Revenue"),
    )
    # TODO: We __might__ be able to put this in the previous `with_columns` statement but
    # we would have to do it while defining the `Total Monthly Revenue` column in the `struct`
    # / `apply` function so I chose to just split it into 2 for simplicity
    .with_columns(
        pl.struct([pl.col("Monthly Revenue"), pl.col("Total Monthly Revenue")])
        .apply(lambda x: x["Monthly Revenue"] / x["Total Monthly Revenue"])
        .alias("Monthly Revenue Percentage"),
    )
)

# Verify that each client has 12 entries
# Looks like "Kaufland" only has 11 entries...
clients = (
    monthly_rev_by_client.select("Client name").unique().get_columns()[0].to_list()
)
for c in clients:
    client_monthly_totals = monthly_rev_by_client.filter(
        pl.col("Client name") == c
    ).shape[0]
    with TermColors.with_failures():
        assert (
            monthly_rev_by_client.filter(pl.col("Client name") == c).shape[0] == 12
        ), f"Client [{c}]: monthly totals [{client_monthly_totals}]"


cmap = sns.color_palette("mako", len(clients))
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
# The dictionary will either have the period as a string or the monthly percentage as a float
previous_clients: List[Dict[str, Union[str, float]]] = []
clients.remove("Kaufland")
clients = sorted(clients)
for color_idx, client_name in enumerate(clients):
    client_data = (
        monthly_rev_by_client.filter(pl.col("Client name") == client_name)
        .select(["Period", "Monthly Revenue Percentage"])
        .to_dict()
    )

    # print(client_data.to_dict())
    # client_mrp = client_data["Monthly Revenue Percentage"]
    def compute_bottom_percentages(
        pc: List[Dict[str, Union[str, float]]]
    ) -> List[float]:
        if len(pc) == 0:
            return None
        elif len(pc) == 1:
            return pc[0]["Monthly Revenue Percentage"].to_list()
        else:
            from collections import Counter
            from functools import reduce
            from operator import add

            # print(reduce(add, map(Counter, pc)))
            # counter = Counter(pc[0])
            counter = Counter()

            for client in pc:
                client_values = pl.from_dict(client).to_dicts()
                # print(client_values)
                client_values = {
                    cv_row["Period"]: cv_row["Monthly Revenue Percentage"]
                    for cv_row in client_values
                }
                counter.update(client_values)
                # print(counter)
                # print(pl.from_dict(client).to_dicts())
                # print(
                # reduce(add, map(lambda x: x["Monthly Revenue Percentage"], client))
                # reduce(add, map(lambda x: type(x), client))
                """"
                print(
                    reduce(
                        add,
                        map(lambda k: k["Monthly Revenue Percentage"], client_values),
                    )
                )
                """
            # print(list(counter.values()))
            return_values = list(counter.values())
            return return_values
            # return counter.values()
            # )
            # counter.update
            # print(client.to_dicts())

            # print(pc[0]["Period"])

            # counter.update
            # print(client.to_dicts())

            # print(pc[0]["Period"])

    ax.bar(
        client_data["Period"],
        client_data["Monthly Revenue Percentage"],
        bottom=compute_bottom_percentages(previous_clients),
        color=cmap[color_idx],
    )
    previous_clients.append(client_data)

print(
    monthly_rev_by_client.filter(pl.col("Period") == "01/01/2015")
    .select("Monthly Revenue Percentage")
    .sum()
)

# Shrink current axis by 15%
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * (1 - 0.15), box.height])
ax.legend(clients, loc="center left", bbox_to_anchor=(1, 0.5))
month_labels = monthly_rev_by_client.get_columns()[0].to_list()
print(ax.get_yticks())
print(ax.get_yticklabels())
ax.set_xticklabels(month_labels, rotation=40, ha="right", fontsize=8)
plt.show()
