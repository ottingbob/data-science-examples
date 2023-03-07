from pathlib import Path

import pandas as pd
import polars as pl

# Going to have an example working with:
# - how to elaborate raw extractions of data
# - how to work efficiently with large quantities of data
# - how to organize data from three different layers
# - how to build a P&L statement from scratch
# - how to create a graphical representation of the company's P&L
case_study_file = Path("financial-analyst-course-2023/case-study-data.xlsm")


def read_fy_from_sheet(sheet_name: str) -> pl.DataFrame:
    fy_pd = pd.read_excel(
        case_study_file,
        sheet_name=sheet_name,
        header=3,
        usecols=range(1, 6),
        names=[
            "P&L account",
            "Partner company",
            "Name of partner company",
            "Amounts",
            "Account number",
        ],
    )
    # Update `Partner company` column to str data type otherwise conversion to
    # polars will blow up...
    fy_pd = fy_pd.astype({"Partner company": str})
    fy_pl = pl.from_pandas(fy_pd)

    # Remove `Total` rows from the `Partner company` column
    fy_pl: pl.DataFrame = fy_pl.filter(pl.col("Partner company") != "Total")

    # Create a `Code` column that combines P&L account and Partner company
    fy_pl = fy_pl.with_columns(
        pl.concat_str(
            [
                pl.col("Account number"),
                pl.col("Partner company"),
            ]
        ).alias("Code")
    )
    return fy_pl


fy_2016 = read_fy_from_sheet(sheet_name="1.1 FY2016")
print(fy_2016)
fy_2017 = read_fy_from_sheet(sheet_name="1.2 FY2017")
print(fy_2017)
fy_2018 = read_fy_from_sheet(sheet_name="1.3 FY2018")
print(fy_2018)

# Create a `Database`: This means just combine data across the three years given
# from the different sheets of the excel document
# We need to make sure that the codes are unique in the final dataframe
fy_16_17_18 = pl.concat([fy_2016, fy_2017, fy_2018], rechunk=True)

# Get unique `Code`
print(fy_16_17_18.shape[0])
unique_codes = fy_16_17_18.select(pl.col("Code")).unique()
print(unique_codes.shape[0])

# For each of the codes add in associated `P&L account`, `Partner company`,
# `Name of partner company`, and amounts for FY16, FY17, FY18
fy_16_17_18.drop_in_place("Amounts")
fy_16_17_18 = fy_16_17_18.unique(subset=["Code"])


def get_year_amounts(
    summary_df: pl.DataFrame, join_df: pl.DataFrame, year_col_name: str
) -> pl.DataFrame:
    join_df_unique = join_df.unique(subset=["Code"])
    return summary_df.with_columns(
        summary_df.with_columns(summary_df["Code"])
        .join(join_df_unique, on="Code", how="outer")
        .with_columns(pl.all().fill_null(0))
        .select("Amounts")
        .apply(lambda a: a[0] * -1)
        .rename({"apply": year_col_name})
        # .alias(year_col_name)
        # .rename({"Amounts": year_col_name})
    )


fy_16_17_18 = get_year_amounts(fy_16_17_18, fy_2016, "FY16")
fy_16_17_18 = get_year_amounts(fy_16_17_18, fy_2017, "FY17")
fy_16_17_18 = get_year_amounts(fy_16_17_18, fy_2018, "FY18")
pl.Config.set_tbl_rows(fy_16_17_18.shape[0])
print(fy_16_17_18)

# Verify our joins were correct by ensuring the value `Net income/(loss)` for
# `P&L Account` is equal to the difference of the sum of all other amounts for a given
# fiscal year
# FIXME: The numbers don't add up =(
#   I think this could be related to the `join_df_unique` on the FY17 & FY18 columns
no_net_income_loss = fy_16_17_18.filter(pl.col("P&L account") != "Net income/(loss)")
fy_sums = no_net_income_loss.select(
    [pl.col("FY16"), pl.col("FY17"), pl.col("FY18")]
).sum()
print(fy_sums)
net_income_loss = fy_16_17_18.filter(
    pl.col("P&L account") == "Net income/(loss)"
).select([pl.col("FY16"), pl.col("FY17"), pl.col("FY18")])
print(net_income_loss)
