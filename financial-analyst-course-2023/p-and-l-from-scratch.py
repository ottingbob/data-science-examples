from pathlib import Path
from typing import Any, Dict, List

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

# Here is an example of a code with more than 1 record
print(fy_2017.filter(pl.col("Code") == "2042000000111101").select("Amounts"))
print(fy_2018.filter(pl.col("Code") == "2042000000111101").select("Amounts"))


def get_year_amounts(
    summary_df: pl.DataFrame, join_df: pl.DataFrame, year_col_name: str
) -> pl.DataFrame:
    # To perform an equivalent of a `SUMIF` we do this in 2 operations...
    # Get the Code / year_col_name aggregation over `Amounts` in 1 frame:
    agg_df = (
        summary_df.with_columns(summary_df["Code"])
        .join(join_df, on="Code", how="left")
        .with_columns(pl.col("Amounts").fill_null(0))
        .groupby("Code")
        .agg(pl.col("Amounts").sum())
        .rename({"Amounts": year_col_name})
    )

    # Now concatenate the results in the original summary frame:
    return summary_df.with_columns(pl.all()).join(agg_df, on="Code", how="left")


fy_16_17_18 = get_year_amounts(fy_16_17_18, fy_2016, "FY16")
fy_16_17_18 = get_year_amounts(fy_16_17_18, fy_2017, "FY17")
fy_16_17_18 = get_year_amounts(fy_16_17_18, fy_2018, "FY18")
pl.Config.set_tbl_rows(fy_16_17_18.shape[0])
print(fy_16_17_18.sort("Code"))


"""
# UPDATE: This has been fixed!!! The issue was not applying an aggregation over the
# columns that have more than 2 results for a given code.
#
# Leaving this example below here since the `pl.coalesce` feature is kinda nifty
# for doing a row update in place
#
# This is a hack to fix the numbers manually...
fy_fix_data = pl.DataFrame(
    {"Code": "2042000000111101", "FY17": -2057298.04, "FY18": -1709688.643}
)
fy_fix = (
    fy_16_17_18.join(fy_fix_data, on="Code", how="left")
    .with_columns(
        [
            pl.coalesce([pl.col("FY17_right"), pl.col("FY17")]).alias("FY17"),
            pl.coalesce([pl.col("FY18_right"), pl.col("FY18")]).alias("FY18"),
        ]
    )
    .drop("FY17_right", "FY18_right")
)
print("UPDATED DF:\n", fy_fix)
fy_16_17_18 = fy_fix
"""

# Verify our joins were correct by ensuring the value `Net income/(loss)` for
# `P&L Account` is equal to the difference of the sum of all other amounts for a given
# fiscal year
no_net_income_loss = fy_16_17_18.filter(pl.col("P&L account") != "Net income/(loss)")
fy_sums = no_net_income_loss.select(
    [pl.col("FY16"), pl.col("FY17"), pl.col("FY18")]
).sum()
print(fy_sums)
net_income_loss = fy_16_17_18.filter(
    pl.col("P&L account") == "Net income/(loss)"
).select([pl.col("FY16"), pl.col("FY17"), pl.col("FY18")])
print(net_income_loss)

for col in ["FY16", "FY17", "FY18"]:
    print(f"{col} sum - net income:", fy_sums[col].sum() - net_income_loss[col].sum())

# Associate `mappings` or categories to our related `P&L account` entries
# So in the lecture they recommend the following ones but in the example these do not match
# up...
"""
mapping = {
    "Revenue": [],
    "Cost of goods sold": [],
    "Operating expenses": [],
    "D&A": [],
    "Interest expenses": [],
    "Extraordinary items": [],
    "Taxes": [],
}
"""
# SO instead rolling with the mappings from the example:
mapping_categories = {
    # Not sure why these 5 were in the example sheet as well...
    "EBITDA": [],
    "EBIT": [],
    "EBT": [],
    "Gross margin": [],
    "Total revenues": [],
    "Net Sales": ["Core business revenues"],
    "Other revenues": ["Other revenues"],
    "Recharges": ["Corporate recharges"],
    "Direct costs": ["Direct costs"],
    "Other operating expenses": [
        "Freight outbound expenses",
        "R&D expenses",
        "Marketing expenses",
        "Software&IT",
        "Charges and contributions",
        "Insurance expenses",
        "Utility expenses",
        "Legal expenses",
        "Misc costs",
        "Consulting fees",
        "Misc extraordinary expenses",
        "Utility charges",
        "Concession fees other",
        # "Travel expenses",
        "Other operative currency differences",
        "Property tax",
        "Operating expenses for utilities",
        "Reimbursements+compensation for damages",
        "Repairs/Maintenance costs",
    ],
    "Personnel expenses": [
        "Wages and salaries",
        "Pension contributions",
        "Severance indemnity contribution",
        "Other personnel expenses",
    ],
    "Leasing": ["Leasings"],
    "Services": ["Service expenses"],
    "Travel costs": ["Travel expenses"],
    "Other income": ["Other income"],
    "Capitalized costs": ["Capitalized costs", "Capitalized interest"],
    "D&A": ["D&A"],
    "Extraordinary items": [
        "Non-recurring costs",
        "Gains from disposal of PP&E",
        "Losses fr disposal of PPE",
        "Misc extraordinary expenses",
        "Impairment of participation",
    ],
    "Financial items": ["Interest income", "Interest expenses"],
    "Taxes": ["Current taxes", "Regional taxes", "Deferred taxes"],
    "Net Income": ["Net income/(loss)"],
}


def apply_mapping(*args):
    account = args[0]
    for category, category_values in mapping_categories.items():
        if account in category_values:
            return category
    return ""


fy_16_17_18 = fy_16_17_18.with_columns(
    pl.col("P&L account").apply(lambda a: apply_mapping(a)).alias("Mapping")
)
print(fy_16_17_18)

# Now organize the P&L by putting the mapping categories in the correct order
pl_categories = [
    # Revenues on top
    "Net Sales",
    "Other revenues",
    "Recharges",
    # The sum of these will give us:
    "Total revenues",
    # Next is costs
    "Direct costs",
    # Difference between total revenues and direct costs are:
    "Gross margin",
    # Operating expenses:
    "Other operating expenses",
    "Personnel expenses",
    "Leasing",
    "Services",
    "Travel costs",
    "Other income",
    "Capitalized costs",
    # Earnings before interests, taxes, depreciation, and amoritization:
    "EBITDA",
    # Depreciation & Amoritization:
    "D&A",
    # Earnings before interests and taxes:
    "EBIT",
    # Next items:
    "Financial items",
    "Extraordinary items",
    # Earnings before taxes:
    "EBT",
    # Finally taxes:
    "Taxes",
    # Final result:
    "Net Income",
]

# Now we need to populate the P&L for FY16 / 17 / 18
pl_headers = ["EUR in millions", "FY16", "FY17", "FY18"]
pl_df = pl.DataFrame({k: pl_categories for k in pl_headers})
print(pl_df)


def compute_aggregate_pl_col(
    year_pl_values: pl.DataFrame,
    sum_cols: List[str],
    year_col: str,
    agg_col: str,
) -> pl.DataFrame:
    year_total_revenue = (
        year_pl_values.filter(pl.col("Mapping").is_in(sum_cols))
        .select(pl.col(year_col).sum().alias(agg_col))
        .to_dict()
        .get(agg_col, [0])[0]
    )
    return pl.concat(
        [
            year_pl_values,
            pl.DataFrame({"Mapping": agg_col, year_col: year_total_revenue}),
        ]
    )


def compute_yearly_pl_values(year_col: str) -> pl.DataFrame:
    fy_pl_values = (
        fy_16_17_18.filter(pl.col("Mapping").is_in(pl_categories))
        .select(year_col, "Mapping")
        .groupby("Mapping")
        .agg(pl.col(year_col).sum())
        # We are in millions and need to flip the sign
        .with_columns(pl.col(year_col).apply(lambda a: (a * -1) / 1_000_000))
    )

    # Compute `Total Revenues`, `Gross Margin`, `EBITDA`,
    # `EBIT` and `EBT`
    fy_pl_values = compute_aggregate_pl_col(
        fy_pl_values,
        ["Net Sales", "Other revenues", "Recharges"],
        year_col,
        "Total revenues",
    )
    fy_pl_values = compute_aggregate_pl_col(
        fy_pl_values,
        ["Total revenues", "Direct costs"],
        year_col,
        "Gross margin",
    )
    fy_pl_values = compute_aggregate_pl_col(
        fy_pl_values,
        [
            "Gross margin",
            "Other operating expenses",
            "Personnel expenses",
            "Leasing",
            "Services",
            "Travel costs",
            "Other income",
            "Capitalized costs",
        ],
        year_col,
        "EBITDA",
    )
    fy_pl_values = compute_aggregate_pl_col(
        fy_pl_values,
        ["EBITDA", "D&A"],
        year_col,
        "EBIT",
    )
    fy_pl_values = compute_aggregate_pl_col(
        fy_pl_values,
        ["EBIT", "Financial items", "Extraordinary items"],
        year_col,
        "EBT",
    )
    return fy_pl_values


fy16_pl_values = compute_yearly_pl_values("FY16")
fy17_pl_values = compute_yearly_pl_values("FY17")
fy18_pl_values = compute_yearly_pl_values("FY18")

# Join all the values to make the final P&L
pl_df = (
    pl_df.with_columns(
        [pl.col("EUR in millions"), pl.col("FY16"), pl.col("FY17"), pl.col("FY18")]
    )
    .join(
        fy16_pl_values,
        left_on="EUR in millions",
        right_on="Mapping",
        how="left",
    )
    .join(
        fy17_pl_values,
        left_on="EUR in millions",
        right_on="Mapping",
        how="left",
    )
    .join(
        fy18_pl_values,
        left_on="EUR in millions",
        right_on="Mapping",
        how="left",
    )
    .with_columns(
        [
            pl.coalesce([pl.col("FY16_right"), pl.col("FY16")]).alias("FY16"),
            pl.coalesce([pl.col("FY17_right"), pl.col("FY17")]).alias("FY17"),
            pl.coalesce([pl.col("FY18_right"), pl.col("FY18")]).alias("FY18"),
        ]
    )
    .drop(["FY16_right", "FY17_right", "FY18_right"])
)
print(pl_df)


def apply_fy_percent_var(*args):
    args = args[0]
    return f"{((float(args[1]) / float(args[0])) - 1) * 100:.2f}%"


# Now calculate percentage variation between FY16-FY17 & FY17-FY18
pl_df = pl_df.with_columns(
    pl_df.select([pl.col("FY16"), pl.col("FY17")])
    .apply(lambda a: apply_fy_percent_var(a))
    .rename({"apply": "Var% FY16-FY17"})
).with_columns(
    pl_df.select([pl.col("FY17"), pl.col("FY18")])
    .apply(lambda a: apply_fy_percent_var(a))
    .rename({"apply": "Var% FY17-FY18"})
)
print(pl_df)


# Calculate KPIs such as GM% EBITDA% and EBIT%
# GM% = GM / Total Revenues
# EBITDA% = EBITDA / Total Revenues
# EBIT% = EBIT / Total Revenues
def apply_gross_margin_percent(*args):
    print(args)
    print(type(args[0][0]))
    # return ("Gross Margin %", 0, 0, 0)
    return args[0]


d: List[Dict[str, Any]] = (
    pl_df.filter(
        pl.any(pl.col("*").is_in(["Gross margin", "Total revenues", "EBITDA", "EBIT"]))
    )
    # Using a struct the row gets passed to the apply function as a dict
    # .select(
    #   pl.struct(["EUR in millions", "FY16", "FY17", "FY18"])
    # )
    # .apply(apply_gross_margin_percent)
    .select(
        pl.col("EUR in millions"),
        pl.col("FY16"),
        pl.col("FY17"),
        pl.col("FY18")
        # pl.struct(["EUR in millions", "FY16", "FY17", "FY18"])
    ).to_dicts()
)

# TODO: Get rekt on dict comprehension:
# total_revenues = row if "Total revenues" in row.values() else {} for row in d
# total_revenues =  for row in d row if "Total revenues" in row.values() else {}
# total_revenues = {row.items() for row in d if "Total revenues" in list(row.values())}
for row in d:
    if "Total revenues" in row.values():
        total_revenues = row
    elif "Gross margin" in row.values():
        gross_margins = row
    elif "EBITDA" in row.values():
        ebitda = row
    elif "EBIT" in row.values():
        ebit = row


def calc_percentage(kpi: str, tr: str) -> str:
    return f"{(float(kpi) / float(tr)) * 100:.2f}%"


def calculate_percentage_kpi(
    kpi_name: str, kpi_row: Dict[str, str], total_revenues: Dict[str, str]
):
    return {
        "EUR in millions": kpi_name,
        "FY16": calc_percentage(kpi_row["FY16"], total_revenues["FY16"]),
        "FY17": calc_percentage(kpi_row["FY17"], total_revenues["FY17"]),
        "FY18": calc_percentage(kpi_row["FY18"], total_revenues["FY18"]),
        "Var% FY16-FY17": "",
        "Var% FY17-FY18": "",
    }


gm_dict = calculate_percentage_kpi("Gross margin %", gross_margins, total_revenues)
ebitda_dict = calculate_percentage_kpi("EBITDA %", ebitda, total_revenues)
ebit_dict = calculate_percentage_kpi("EBIT %", ebit, total_revenues)
kpi_row = {k: "" for k in pl_df.columns}
kpi_row["EUR in millions"] = "KPIs"
pl_df = pl.concat(
    [
        pl_df,
        pl.DataFrame(kpi_row),
        pl.DataFrame(gm_dict),
        pl.DataFrame(ebitda_dict),
        pl.DataFrame(ebit_dict),
    ]
)
print(pl_df)
