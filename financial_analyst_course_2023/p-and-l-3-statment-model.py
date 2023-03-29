import operator
import os
from pathlib import Path
from typing import Callable

import pandas as pd
import polars as pl

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
print(agg_df_for_pandl)
