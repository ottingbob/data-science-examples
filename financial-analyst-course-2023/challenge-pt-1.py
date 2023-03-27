import calendar
import operator
import os
import re
from contextlib import contextmanager
from datetime import datetime
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
    OKBLUE = "\033[34m"
    OKWHITE = "\033[37m"
    OKCYAN = "\033[96m"
    # OKGREEN = "\033[92m"
    OKGREEN = "\033[32m"
    WARNING = "\033[93m"
    # FAIL = "\033[91m"
    FAIL = "\033[31m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @classmethod
    def df_header(cls, header_text: str) -> str:
        return f"\n{cls.OKGREEN}{cls.BOLD}{header_text}{cls.ENDC}\n"

    @classmethod
    def print_df_with_colors(cls, text: str):
        # Compile regex patterns
        header_row = re.compile(r"^\\|\s+?(Calculation)")
        profit_row = re.compile(r"^\\|\s+?(Monthly Gross Profit)")
        cogs_row = re.compile(r"^\\|\s+?(Monthly Cogs)")
        rev_row = re.compile(r"^\\|\s+?(Monthly Revenue)")
        number_pattern = re.compile(r"([0-9]+\.[0-9]+)")
        date_pattern = re.compile(r"([0-9]{2}/[0-9]{2}/[0-9]{4})")

        def color_line(
            line_text: str,
            row_label: str,
            color_format: str,
            values_regex: re.Pattern = number_pattern,
        ) -> str:
            line_text = line_text.replace(row_label, color_format % (row_label))
            for value in values_regex.findall(line_text):
                line_text = line_text.replace(value, color_format % (value))
            return line_text

        # Apply regex patterns over lines
        for idx, line_text in enumerate(text.splitlines()):
            # exclude shape row
            if line_text.startswith("shape:"):
                continue

            if header_label := header_row.search(line_text):
                line_text = color_line(
                    line_text,
                    header_label.group(),
                    f"{cls.OKBLUE}{cls.BOLD}%s{cls.ENDC}",
                    date_pattern,
                )
            elif rev_label := rev_row.search(line_text):
                line_text = color_line(
                    line_text,
                    rev_label.group(),
                    f"{cls.OKWHITE}{cls.BOLD}%s{cls.ENDC}",
                )
            elif cogs_label := cogs_row.search(line_text):
                line_text = color_line(
                    line_text,
                    cogs_label.group(),
                    f"{cls.FAIL}{cls.BOLD}%s{cls.ENDC}",
                )
            elif profit_label := profit_row.search(line_text):
                line_text = color_line(
                    line_text,
                    profit_label.group(),
                    f"{cls.OKGREEN}{cls.BOLD}%s{cls.ENDC}",
                )
            print(line_text)

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
data_file = "93.+Course-challenge-before-solutions.xlsx"
course_challenge_file = Path(str(Path(__file__).parent) + os.sep + data_file)

rev_col = "Revenue ($ 000')"
cogs_col = "Cogs ($ 000')"
pd_df = pd.read_excel(
    course_challenge_file,
    sheet_name="Data",
    # Rows start from `0` so `A` == `0`
    header=2,
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
print(
    f"{TermColors.OKWHITE}{TermColors.BOLD}Annual Revenue 2015: {TermColors.OKGREEN}{annual_revenue:.2f}{TermColors.ENDC}"
)
print(
    f"{TermColors.OKWHITE}{TermColors.BOLD}Annual Cogs 2015: {TermColors.FAIL}{annual_cogs:.2f}{TermColors.ENDC}"
)
print(
    f"{TermColors.OKWHITE}{TermColors.BOLD}Annual Gross Profit 2015: {TermColors.OKGREEN}{annual_gross_profit:.2f}{TermColors.ENDC}"
)

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

rev_cogs_by_month = (
    monthly_rev_cogs
    # Drop period and make period the column names
    .select(pl.exclude("Period")).transpose(
        # Include the header as a separate column
        include_header=True,
        # Add the periods as the column names
        column_names=monthly_rev_cogs.with_columns(
            pl.col("Period").cast(pl.Date).dt.strftime("%m/%d/%Y")
        )
        .get_columns()[0]
        .to_list(),
        header_name="Calculation",
    )
)
print(
    TermColors.df_header("Monthly development of Revenues & Cogs FY15:"),
)
TermColors.print_df_with_colors(str(rev_cogs_by_month)),


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
# plot_monthly_rev_cogs(smooth_gross_profit=True)

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
clients = (
    monthly_rev_by_client.select("Client name").unique().get_columns()[0].to_list()
)
month_labels = monthly_rev_by_client.get_columns()[0].to_list()
for c in clients:
    client_monthly_totals = monthly_rev_by_client.filter(
        pl.col("Client name") == c
    ).shape[0]
    with TermColors.with_failures():
        # SWEET THIS HAS BEEN FIXED!!! I wasn't reading the first row of data from the spreadsheet
        # when I was importing it =(
        try:
            assert (
                monthly_rev_by_client.filter(pl.col("Client name") == c).shape[0] == 12
            ), f"Client [{c}]: monthly totals [{client_monthly_totals}]"
        # Add in the missing month label to the dataframe
        except AssertionError as ae:
            add_months = set(
                monthly_rev_by_client.filter(pl.col("Client name") == c)
                .select(["Period"])
                .get_columns()[0]
                .to_list()
            ).symmetric_difference(set(month_labels))
            for month in add_months:
                monthly_rev_by_client = monthly_rev_by_client.extend(
                    pl.from_dict(
                        {
                            "Period": month,
                            "Client name": c,
                            "Monthly Revenue": 0.0,
                            "Total Monthly Revenue": 0.0,
                            "Monthly Revenue Percentage": 0.0,
                        }
                    )
                )
            raise ae


def plot_client_monthly_revenue_stacked_bar_chart():
    cmap = sns.color_palette("mako", len(clients))
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    # The dictionary will either have the period as a string or the monthly percentage as a float
    previous_clients: List[Dict[str, Union[str, float]]] = []
    sorted_clients = sorted(clients)
    for color_idx, client_name in enumerate(sorted_clients):
        client_data = (
            monthly_rev_by_client.filter(pl.col("Client name") == client_name)
            .select(["Period", "Monthly Revenue Percentage"])
            .to_dict()
        )

        def compute_bottom_percentages(
            pc: List[Dict[str, Union[str, float]]]
        ) -> List[float]:
            if len(pc) == 0:
                return None
            elif len(pc) == 1:
                return pc[0]["Monthly Revenue Percentage"].to_list()
            else:
                from collections import Counter

                counter = Counter()
                for client in pc:
                    client_values = pl.from_dict(client).to_dicts()
                    client_values = {
                        cv_row["Period"]: cv_row["Monthly Revenue Percentage"]
                        for cv_row in client_values
                    }
                    counter.update(client_values)

                    """"
                    from functools import reduce
                    from operator import add

                    # print(reduce(add, map(Counter, pc)))
                    TODO: This would be cool to try and implement
                    print(
                        reduce(
                            add,
                            map(lambda k: k["Monthly Revenue Percentage"], client_values),
                        )
                    )
                    """
                return_values = list(counter.values())
                return return_values

        ax.bar(
            client_data["Period"],
            client_data["Monthly Revenue Percentage"],
            bottom=compute_bottom_percentages(previous_clients),
            color=cmap[color_idx],
        )
        previous_clients.append(client_data)

    # Shrink current axis by 15% and put legend in the space
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * (1 - 0.15), box.height])
    ax.legend(clients, loc="center left", bbox_to_anchor=(1, 0.5))

    # Change y ticks to percentage values instead of decimals
    ytl = ax.get_yticklabels()
    new_ticks = []
    for tick in ytl:
        t = tick
        t._text = f"{float(t._text) * 100:0.0f}%"
        new_ticks.append(t)

    ax.set_yticklabels(new_ticks)
    ax.set_xticklabels(month_labels, rotation=40, ha="right", fontsize=8)
    plt.title("Revenue breakdown per month stacked by client")
    plt.ylabel("Percent Allocation")
    plt.xlabel("Month")
    plt.show()


# plot_client_monthly_revenue_stacked_bar_chart()

# So what we are actually looking for is a stacked bar chart broken down by
# different client types or the `Type of client` column
# Lets do that below:
mrev_by_client_type = (
    df.with_columns(pl.col("Period"))
    .groupby("Period", "Type of client")
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
client_types = sorted(
    mrev_by_client_type.select("Type of client").unique().get_columns()[0].to_list()
)


def plot_client_type_monthly_revenue_stacked_bar_chart():
    cmap = sns.color_palette("mako", len(client_types))
    plt.figure(figsize=(11, 8))
    ax = plt.subplot(111)

    # The dictionary will either have the period as a string or the monthly percentage as a float
    previous_types: List[Dict[str, Union[str, float]]] = []
    for color_idx, client_type in enumerate(client_types):
        client_type_data = (
            mrev_by_client_type.filter(pl.col("Type of client") == client_type)
            .select(["Period", "Monthly Revenue Percentage"])
            .to_dict()
        )

        def compute_bottom_percentages(
            pc: List[Dict[str, Union[str, float]]]
        ) -> List[float]:
            if len(pc) == 0:
                return None
            elif len(pc) == 1:
                return pc[0]["Monthly Revenue Percentage"].to_list()
            else:
                from collections import Counter

                counter = Counter()
                for client in pc:
                    client_values = pl.from_dict(client).to_dicts()
                    client_values = {
                        cv_row["Period"]: cv_row["Monthly Revenue Percentage"]
                        for cv_row in client_values
                    }
                    counter.update(client_values)

                    """"
                    from functools import reduce
                    from operator import add

                    # print(reduce(add, map(Counter, pc)))
                    TODO: This would be cool to try and implement
                    print(
                        reduce(
                            add,
                            map(lambda k: k["Monthly Revenue Percentage"], client_values),
                        )
                    )
                    """
                return_values = list(counter.values())
                return return_values

        ax.bar(
            client_type_data["Period"],
            client_type_data["Monthly Revenue Percentage"],
            bottom=compute_bottom_percentages(previous_types),
            color=cmap[color_idx],
            width=0.85,
        )
        previous_types.append(client_type_data)

    # Shrink current axis by 10% and put legend in the space
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * (1 - 0.10), box.height])
    ax.legend(client_types, loc="center left", bbox_to_anchor=(1, 0.5))

    # Change y ticks to percentage values instead of decimals
    ytl = ax.get_yticklabels()
    new_ticks = []
    for tick in ytl:
        t = tick
        t._text = f"{float(t._text) * 100:0.0f}%"
        new_ticks.append(t)

    ax.set_yticklabels(new_ticks)

    # Convert the month labels from mm/dd/yyyy to Month-Year
    converted_labels = []
    for ml in month_labels:
        date: datetime = datetime.strptime(ml, "%m/%d/%Y")
        final_label = f"{calendar.month_abbr[date.month]}-{date.year}"
        converted_labels.append(final_label)
    ax.set_xticklabels(converted_labels, rotation=40, ha="right", fontsize=8)

    # Add labels to bars themselves
    for idx in range(len(ax.containers)):
        ax.bar_label(
            ax.containers[idx],
            labels=[f"{label * 100:0.2f}%" for label in ax.containers[idx].datavalues],
            label_type="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )

    plt.title("Revenue by client type as a percentage of Total Revenues")
    plt.ylabel("Percent Allocation")
    plt.xlabel("Month")
    plt.show()


# Plot stacked bar chart for client type as a percentage of total revenue
# plot_client_type_monthly_revenue_stacked_bar_chart()

client_monthly_revenues = mrev_by_client_type.pivot(
    values=["Monthly Revenue", "Monthly Revenue Percentage"],
    index="Type of client",
    columns="Period",
)
# Table with month col and client type row for monthly revenue
print(
    TermColors.df_header("Client monthly revenues:"),
    client_monthly_revenues.select(pl.exclude(r"^.*Percentage.*$")),
)
# Table with month col and client type row for percent of total revenues
print(
    TermColors.df_header("Client monthly revenue percentages:"),
    client_monthly_revenues.select(
        pl.exclude(r"^.*Revenue_[0-9]{2}/[0-9]{2}/[0-9]{4}$")
    ),
)
