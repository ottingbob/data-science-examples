from typing import Tuple

import numpy as np
import numpy_financial as npf
import polars as pl

# Money today is `more valuable` than money tomorrow

# Discounted cash flows and obtaining present value are a way to account for
# the time value of money

# If an investor wants to buy a specific stock they should evaluate the future
# cash flows that the stock would generate, discount the cash flows and then finally
# add the present values:
# Now -- (Future Cash Flow 1 / (1+i)) -- (Future Cash Flow 2 / (1+i)**2)
# Now < PV(1) + PV(2) --> This is a feasible / investment with value

# Net Present Value = PV(1) + ... + PV(n) - Initial Investment

interest_rate = 0.1
discounting_cash_flows = pl.DataFrame(
    {"Years": list(range(6)), "Cash Flow": [-500, 30, 120, 200, 120, 120]}
)
print(discounting_cash_flows)


def present_value(row: Tuple[int, int]):
    years, cash_flow = row
    return cash_flow / (1 + interest_rate) ** years


discounting_cash_flows = discounting_cash_flows.with_columns(
    pl.struct(pl.col(["Years", "Cash Flow"]))
    .apply(lambda x: present_value(x.values()))
    .alias("Present Value")
)
print(discounting_cash_flows)

net_present_value = discounting_cash_flows.select(pl.col("Present Value")).sum()
print(net_present_value)
print(npf.npv(interest_rate, discounting_cash_flows.get_column("Cash Flow").to_list()))
"""
Since the sum is negative, the project is not feasible
┌───────────────┐
│ Present Value │
│ ---           │
│ f64           │
╞═══════════════╡
│ -66.818585    │
└───────────────┘
"""


# Internal rate of return is a discount rate used for measuring the profitability
# of a potential investment
# The effective rate of return makes the net present value of all cash flows from an
# investment equal to 0
# IRR calculates the rate we would have had IF the NPV was equal to 0

irr = npf.irr(discounting_cash_flows.get_column("Cash Flow").to_list())
print("Internal rate of return:", irr)


def irr_cash_flow(row: Tuple[int, int]):
    years, cash_flow = row
    return cash_flow / (1 + irr) ** years


discounting_cash_flows = discounting_cash_flows.with_columns(
    pl.struct(pl.col(["Years", "Cash Flow"]))
    .apply(lambda x: irr_cash_flow(x.values()))
    .alias("IRR Cash Flow")
)
print(discounting_cash_flows)

# This is still a bad project because the IRR < interest rate on the financing so
# we will obtain returns that cannot cover the cost of the financing

# Payment calculates the constant monthly payment that is necessary to extinguish
# a loan in `n` periods
# The numpy_financial function for this is `pmt`

# Our example will use a loan schedule to calculate monthly payment:
num_periods = 10 * 12
annual_interest_rate = 0.03
monthly_interest_rate = annual_interest_rate / 12
loan_amount = 300_000

monthly_payment = abs(
    npf.pmt(monthly_interest_rate, num_periods, loan_amount, when="end")
)
print(monthly_payment)

# Let's separate principal from interest etc.
payment_num = list(range(1, num_periods + 1))
payments = [round(monthly_payment, 2) for _ in range(num_periods)]
# Interest = Residual Debt * monthly_interest_rate
"""
interest = [
    round(((loan_amount - (payment * monthly_payment)) * monthly_interest_rate), 2)
    for payment in range(num_periods)
]
"""
interest = npf.ipmt(
    monthly_interest_rate, np.arange(1 * num_periods) + 1, num_periods, loan_amount
)
interest = np.round(interest, 2) * -1
# Principal + Interest = Payment
principal = [
    round(monthly_payment - interest_amount, 2) for interest_amount in interest
]
# Residual debt(t) = residual debt(t - 1) - principal(t)
residual_debt = [round(loan_amount - sum(principal[:i]), 2) for i in payment_num]

loan_schedule = pl.DataFrame(
    {
        "Period": payment_num,
        "Payment": payments,
        "Interest": interest,
        "Principal": principal,
        "Residual Debt": residual_debt,
    }
)
print(loan_schedule)
