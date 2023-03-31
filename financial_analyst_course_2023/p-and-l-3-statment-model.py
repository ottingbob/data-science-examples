import math
import operator
import os
from pathlib import Path
from typing import Callable, List

import pandas as pd
import polars as pl
from financial_analyst_course_2023.term_colors import TermColors

# First start by getting data into a dataframe
data_file = "106.+Exerxise-before.xlsx"
course_challenge_file = Path(str(Path(__file__).parent) + os.sep + data_file)

pd_df = pd.read_excel(
    course_challenge_file,
    sheet_name="Data Source",
    # Rows start from `0` so `A` == `0`
    header=2,
    usecols=range(1, 4),
    names=[
        "Date",
        "Financials",
        "Amount",
    ],
)
df = pl.from_pandas(pd_df)

# Convert date column to date datatype and then create year column
df = df.with_columns(pl.col("Date").str.strptime(pl.Date, "%m.%d.%Y")).with_columns(
    pl.col("Date").dt.year().alias("Year")
)

# print(df.select("Financials").unique().get_columns()[0].to_list())
# ['Revenue - Supermarkets', 'Cogs - Supermarkets', 'Revenue - Discounters', 'Cogs - Discounters', 'Revenue - Grocery', 'Cogs - Grocery', 'Revenue - Retail', 'Cogs - Retail', 'Operating expenses', 'D&A', 'Interest expenses', 'Taxes', 'Other revenue']


# Split out this column to either be Cogs / Revenue / or what is already currently in there
def apply_mapping_col(financials_col: str):
    if financials_col.startswith("Revenue") or financials_col.endswith("revenue"):
        return "Revenue"
    elif financials_col.startswith("Cogs"):
        return "Cogs"
    return financials_col


df = df.with_columns(
    pl.col("Financials").apply(lambda x: apply_mapping_col(x)).alias("Mapping")
)

# Now get the sum over the mapping fields for the corresponding years
df_for_pandl = df.select(["Year", "Mapping", "Amount"])
pl.Config.set_tbl_rows(21)
agg_df_for_pandl = (
    df_for_pandl.groupby("Year", "Mapping")
    .agg([pl.col("Amount").sum().alias("Mapping Year Sum")])
    .sort("Year", "Mapping")
)
agg_df_for_pandl = agg_df_for_pandl.pivot(
    values=["Mapping Year Sum"], index="Mapping", columns="Year"
)


def compute_agg_pl_row(
    agg_df: pl.DataFrame,
    row_name: str,
    operation: Callable,
    primary_col: str,
    secondary_col: str,
) -> pl.DataFrame:
    row_data = agg_df.filter(
        pl.col("Mapping").is_in([primary_col, secondary_col])
    ).to_dicts()
    primary_row = row_data[0] if primary_col in row_data[0].values() else row_data[1]
    secondary_row = (
        row_data[0] if secondary_col in row_data[0].values() else row_data[1]
    )
    return pl.concat(
        [
            agg_df,
            pl.DataFrame(
                {
                    "Mapping": row_name,
                    "2014": operation(primary_row["2014"], secondary_row["2014"]),
                    "2015": operation(primary_row["2015"], secondary_row["2015"]),
                    "2016": operation(primary_row["2016"], secondary_row["2016"]),
                }
            ),
        ]
    )


"""
rev_cogs = agg_df_for_pandl.filter(
    pl.col("Mapping").is_in(["Cogs", "Revenue"])
).to_dicts()
cogs_row = rev_cogs[0] if "Cogs" in rev_cogs[0].values() else rev_cogs[1]
rev_row = rev_cogs[0] if "Revenue" in rev_cogs[0].values() else rev_cogs[1]
agg_df_for_pandl = pl.concat(
    [
        agg_df_for_pandl,
        pl.DataFrame(
            {
                "Mapping": "Gross Profit",
                "2014": rev_row["2014"] - cogs_row["2014"],
                "2015": rev_row["2015"] - cogs_row["2015"],
                "2016": rev_row["2016"] - cogs_row["2016"],
            }
        ),
    ]
)
"""
agg_df_for_pandl = compute_agg_pl_row(
    agg_df_for_pandl, "Gross Profit", operator.sub, "Revenue", "Cogs"
)
agg_df_for_pandl = compute_agg_pl_row(
    agg_df_for_pandl, "EBITDA", operator.sub, "Gross Profit", "Operating expenses"
)
agg_df_for_pandl = compute_agg_pl_row(
    agg_df_for_pandl, "EBIT", operator.sub, "EBITDA", "D&A"
)
agg_df_for_pandl = compute_agg_pl_row(
    agg_df_for_pandl, "EBT", operator.sub, "EBIT", "Interest expenses"
)
agg_df_for_pandl = compute_agg_pl_row(
    agg_df_for_pandl, "Net Income", operator.sub, "EBT", "Taxes"
)
mapping_sort = {
    "Revenue": 0,
    "Cogs": 1,
    "Gross Profit": 2,
    "Operating expenses": 3,
    "EBITDA": 4,
    "D&A": 5,
    "EBIT": 6,
    "Interest expenses": 7,
    "EBT": 8,
    "Taxes": 9,
    "Net Income": 10,
}
agg_df_for_pandl = (
    agg_df_for_pandl.with_columns(
        pl.col("Mapping").apply(lambda x: mapping_sort.get(x, 0)).alias("sort")
    )
    .sort("sort")
    .drop("sort")
)
# Add in Var % columns for 14-15 & 15-16
agg_df_for_pandl = agg_df_for_pandl.with_columns(
    pl.struct([pl.col("2014"), pl.col("2015")])
    .apply(lambda x: f"{((x['2015'] / x['2014']) - 1) * 100:0.2f}%")
    .alias("Var % 14-15"),
    pl.struct([pl.col("2015"), pl.col("2016")])
    .apply(lambda x: f"{((x['2016'] / x['2015']) - 1) * 100:0.2f}%")
    .alias("Var % 15-16"),
)
TermColors.print_pandl_with_colors(str(agg_df_for_pandl))

# Read in balance sheet for 2014 / 2015 / 2016
data_file = "111.+Exercise+-+before.xlsx"
course_challenge_file = Path(str(Path(__file__).parent) + os.sep + data_file)

asset_line_items = [
    "Trade Receivables",
    "Inventory",
    "PP&E",
    "Cash",
    "Other assets",
]
liabilities_line_items = [
    "Trade Payables",
    "Provisions",
    "Financial Liabilities",
    "Other liabilities",
    "Equity",
]


def parse_balance_sheet(
    sheet_name: str,
    use_cols: List[int],
    eoy_column_name: str,
) -> pd.DataFrame:
    pd_2014_bs = pd.read_excel(
        course_challenge_file,
        sheet_name=sheet_name,
        header=3,
        usecols=use_cols,
        names=[
            "Category",
            "EOY Balance",
        ],
    ).dropna()
    assets_2014 = (
        pd_2014_bs.loc[pd_2014_bs["Category"].isin(asset_line_items)]
        .drop_duplicates()
        .reset_index()
        .drop(["index"], axis=1)
    )
    assets_2014.loc[len(assets_2014.index)] = [
        "Assets",
        assets_2014["EOY Balance"].sum(),
    ]

    liabilities_2014 = (
        pd_2014_bs.loc[pd_2014_bs["Category"].isin(liabilities_line_items)]
        .drop_duplicates()
        .reset_index()
        .drop(["index"], axis=1)
    )
    liabilities_2014.loc[len(liabilities_2014.index)] = [
        "Liabilities & Equity",
        liabilities_2014["EOY Balance"].sum(),
    ]

    pd_2014_bs = (
        pd.concat([assets_2014, liabilities_2014])
        .reset_index()
        .drop(["index"], axis=1)
        .rename(columns={"EOY Balance": eoy_column_name})
    )
    return pd_2014_bs


pd_2014_bs = parse_balance_sheet(
    sheet_name="BS 2014", use_cols=[1, 3], eoy_column_name="31-Dec-2014"
)

pd_2015_bs = parse_balance_sheet(
    sheet_name="BS 2015", use_cols=[1, 3], eoy_column_name="31-Dec-2015"
)

pd_2016_bs = parse_balance_sheet(
    sheet_name="BS 2016", use_cols=[1, 4], eoy_column_name="31-Dec-2016"
)

pd_bs = pd_2014_bs.merge(pd_2015_bs, on="Category", how="inner")
pd_bs = pd_bs.merge(pd_2016_bs, on="Category", how="inner")

# Ensure assets and liabilities equal out to `0`
assets = pd_bs.loc[pd_bs["Category"] == "Assets"].to_dict(orient="list")
liabilities = pd_bs.loc[pd_bs["Category"] == "Liabilities & Equity"].to_dict(
    orient="list"
)
del assets["Category"]
del liabilities["Category"]
assert assets.keys() == liabilities.keys()
for k in assets.keys():
    assert math.isclose(assets[k][0], liabilities[k][0])

print(pd_bs)

# TODO: Need to work on the Cash Flow Statement
# TODO: Need to work on the Income Statement
