from itertools import product
from typing import Tuple

import polars as pl

price = 300_000
annual_intrest_rate = 0.05
monthly_intrest_rate = annual_intrest_rate / 12
load_duration_years = 20
number_of_months = 20 * 12


# https://superuser.com/questions/871404/what-would-be-the-the-mathematical-equivalent-of-this-excel-formula-pmt
def pmt():
    monthly_payment = (price * monthly_intrest_rate) / (
        1 - (1 + monthly_intrest_rate) ** (-number_of_months)
    )
    return monthly_payment


print(pmt())


# Interest rate values data table example
interest = 0.1
financing = 1_000
years = 5

to_be_paid = financing * (1 + interest) ** years
print(to_be_paid)

variable_interest_rates = [0.1, 0.11, 0.12, 0.13]
number_of_periods = [2, 3, 4, 5]
print(list(product(variable_interest_rates, number_of_periods)))


# def compute_payment(interest_rate, num_periods):
def compute_payment(row: Tuple[float, int]):
    interest_rate, num_periods = row
    return financing * (1 + interest_rate) ** num_periods


df_interest_rates = pl.DataFrame({"InterestRates": variable_interest_rates})
df_num_periods = pl.DataFrame({"NumberOfPeriods": number_of_periods})
df_cross: pl.DataFrame = df_interest_rates.join(df_num_periods, how="cross")
df_cross = df_cross.with_columns(
    pl.struct(pl.col(["InterestRates", "NumberOfPeriods"]))
    .apply(lambda x: compute_payment(x.values()))
    .alias("Payment")
)
# Shape is a tuple of (rows, cols)
pl.Config.set_tbl_rows(df_cross.shape[0])
print(df_cross)
