import os
from functools import partial
from pathlib import Path
from typing import Any, List, Tuple

import polars as pl

# First start by getting data into a dataframe
data_file = "94.+Exercise-Build-an-FMCG-Model-Lecture-before.xlsx"
course_challenge_file = Path(str(Path(__file__).parent) + os.sep + data_file)

df = pl.read_excel(
    course_challenge_file,
    sheet_name="Data",
    xlsx2csv_options={"skip_empty_lines": True},
    read_csv_options={"has_header": True, "null_values": ["x"]},
).drop_nulls()

# TODO: Can optionally save it to a non-excel file to enable lazy loading on
# a subsequent run but will save that for another time...

# Print columns and datatypes:
print(list(zip(df.columns, df.dtypes)))

# We need to add in a date & month column
df_with_dates = df.with_columns(
    pl.col("Period").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m").alias("Date"),
).with_columns(
    pl.col("Date").apply(lambda x: x.month).alias("Month"),
    pl.col("Date").apply(lambda x: x.year).alias("Year"),
)


def get_col_sum_by_year(df: pl.DataFrame, column_name: str, year: int) -> float:
    return round(
        df.filter(pl.col("Year") == year).select([column_name]).sum().item() / 1_000, 2
    )


def get_col_sums_over_years(df: pl.DataFrame, column_name: str) -> Tuple[float, float]:
    return (
        get_col_sum_by_year(df, column_name, 2015),
        get_col_sum_by_year(df, column_name, 2016),
    )


def concat_lists(*lists: List[List[Any]]) -> List[Any]:
    result = []
    # TODO: return if all lists are not the same length
    if len(lists) < 2 or len(lists[0]) < 1:
        return result

    for i in range(len(lists[0])):
        col_list = []
        for _list in lists:
            col_list.append(_list[i])
        result.append(col_list)
    return result


gcsoy = partial(get_col_sums_over_years, df_with_dates)
volume_2015, volume_2016 = gcsoy("Volume")
gs_15, gs_16 = gcsoy("Gross Sales")
discounts_15, discounts_16 = gcsoy("Discounts")
ns_15, ns_16 = gcsoy("Net Sales")
cogs_15, cogs_16 = gcsoy("Cost of Goods Sold")
dist_15, dist_16 = gcsoy("Distribution")
whs_15, whs_16 = gcsoy("Warehousing")

# Build the model as a dataframe and print to the terminal
print("FDM, Sales & Volume Analysis")
df_table_header = ["USD in 000s", "2015", "2016"]

volume_row = ["Volume", volume_2015, volume_2016]
gsi_row = ["Gross Sales Income", gs_15, gs_16]
discounts_row = ["Discounts", discounts_15, discounts_16]
ns_row = ["Net Sales", ns_15, ns_16]
cogs_row = ["Cost of Goods Sold", cogs_15, cogs_16]
# COGS is already negative
gp_row = ["Gross Profit", ns_15 + cogs_15, ns_16 + cogs_16]
dist_row = ["Distribution", dist_15, dist_16]
whs_row = ["Warehousing", whs_15, whs_16]
# COGS / Dist / WHS are already negative
fdm_row = [
    "Full Delivered Margin",
    sum([ns_15, cogs_15, dist_15, whs_15]),
    sum([ns_16, cogs_16, dist_16, whs_16]),
]
values = concat_lists(
    volume_row,
    gsi_row,
    discounts_row,
    ns_row,
    cogs_row,
    gp_row,
    dist_row,
    whs_row,
    fdm_row,
)

pl.Config.set_tbl_rows(20)
df_table = pl.DataFrame({z[0]: z[1] for z in zip(df_table_header, values)})

# Add in `Variance ABS` and `Variance %` for 15-16
df_table = df_table.with_columns(
    pl.struct(pl.col(["2015", "2016"]))
    .apply(lambda x: x["2016"] - x["2015"])
    .alias("Variance Abs 15-16"),
    pl.struct(pl.col(["2015", "2016"]))
    .apply(lambda x: f"{((x['2016'] / x['2015']) - 1) * 100:0.2f}%")
    .alias("Variance % 15-16"),
)
print(df_table)

# Add in KPIs such as `Gross Profit %` and `FDM %`
print("KPI Analysis")
gp_percent_15 = f"{((ns_15 + cogs_15) / ns_15) * 100:0.2f}%"
gp_percent_16 = f"{((ns_16 + cogs_16) / ns_16) * 100:0.2f}%"
fdm_percent_15 = f"{(sum([ns_15, cogs_15, dist_15, whs_15]) / ns_15) * 100:0.2f}%"
fdm_percent_16 = f"{(sum([ns_16, cogs_16, dist_16, whs_16]) / ns_16) * 100:0.2f}%"

gp_percent_row = ["Gross Profit %", gp_percent_15, gp_percent_16]
fdm_percent_row = ["FDM %", fdm_percent_15, fdm_percent_16]
values = concat_lists(gp_percent_row, fdm_percent_row)
kpi_df = pl.DataFrame({z[0]: z[1] for z in zip(["KPIs", "2015", "2016"], values)})
print(kpi_df)

# Ideally we have ways to filter this data and then regenerate the table above based on
# filters such as client / brand / client type / etc.
