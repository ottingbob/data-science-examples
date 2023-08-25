# Present and Future value of money
#
# Time value of money is $x is worth more today than $x in a years time due
# to interest

# Future value of $x:
#
# Discrete model:
# x(1 + r) ^ n
# Continuous model:
# x(t) = x(0)e ^ (r * t)

from math import exp


def future_discrete_value(value: float, rate: float, years: int) -> float:
    return value * ((1 + rate) ** years)


def present_discrete_value(future_value: float, rate: float, years: int) -> float:
    return future_value / ((1 + rate) ** years)


def future_continuous_value(value: float, rate: float, years: int) -> float:
    return value * exp(rate * years)


def present_continuous_value(future_value: float, rate: float, years: int) -> float:
    return future_value * exp(-rate * years)


# Here are some examples
initial = 100
# 5 percent
interest_rate = 0.05
# 5 years
duration = 5

print("Future discrete value:", future_discrete_value(initial, interest_rate, duration))
print(
    "Present discrete value:", present_discrete_value(initial, interest_rate, duration)
)
print(
    "Future continuous value:",
    future_continuous_value(initial, interest_rate, duration),
)
print(
    "Present continuous value:",
    present_continuous_value(initial, interest_rate, duration),
)
"""
Future discrete value: 127.62815625000003
Present discrete value: 78.35261664684589
Future continuous value: 128.40254166877415
Present continuous value: 77.8800783071405
"""
