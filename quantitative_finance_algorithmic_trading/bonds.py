from math import exp


class ZeroCouponBond:
    def __init__(self, principal: float, maturity: int, interest_rate: float):
        self._principal = principal
        # date to maturity
        self._maturity = maturity
        # market related interest rate (discounting)
        self._interest_rate = interest_rate / 100

    @property
    def principal(self) -> float:
        return self._principal

    @property
    def maturity(self) -> int:
        return self._maturity

    @property
    def interest_rate(self) -> float:
        return self._interest_rate

    # With zero coupon bonds n is the same as the time to maturity
    def present_value(self, initial: float, years: int) -> float:
        return initial / (1 + self.interest_rate) ** years

    def present_continuous_value(self, future_value: float, years: int) -> float:
        return future_value * exp(-self.interest_rate * years)

    def calculate_price(self) -> float:
        return self.present_value(self.principal, self.maturity)

    def calculate_continuous_price(self) -> float:
        return self.present_continuous_value(self.principal, self.maturity)


class CouponBond:
    def __init__(
        self,
        principal: float,
        rate: float,
        maturity: int,
        interest_rate: float,
    ):
        self._principal = principal
        # coupon rate
        self._rate = rate / 100
        # date to maturity
        self._maturity = maturity
        # market related interest rate (discounting)
        self._interest_rate = interest_rate / 100

    @property
    def principal(self) -> float:
        return self._principal

    @property
    def rate(self) -> float:
        return self._rate

    @property
    def maturity(self) -> int:
        return self._maturity

    @property
    def interest_rate(self) -> float:
        return self._interest_rate

    def present_value(self, initial: float, years: int) -> float:
        return initial / (1 + self.interest_rate) ** years

    def present_continuous_value(self, future_value: float, years: int) -> float:
        return future_value * exp(-self.interest_rate * years)

    def calculate_price(self) -> float:
        price = 0

        # discount the present value of coupon payments
        for t in range(1, self.maturity + 1):
            price = price + self.present_value(
                self.principal * self.rate,
                t,
            )

        # discount principle amount
        price = price + self.present_value(self.principal, self.maturity)
        return price

        return self.present_value(self.principal, self.maturity)

    def calculate_continuous_price(self) -> float:
        price = 0

        # discount the present value of coupon payments
        for t in range(1, self.maturity + 1):
            price = price + self.present_continuous_value(
                self.principal * self.rate,
                t,
            )

        # discount principle amount
        price = price + self.present_continuous_value(self.principal, self.maturity)
        return price

        return self.present_continuous_value(self.principal, self.maturity)


if __name__ == "__main__":
    bond = ZeroCouponBond(
        principal=1000,
        maturity=2,
        interest_rate=4,
    )
    print(
        f"Price of the zero coupon bond in dollars: ${round(bond.calculate_price(), 2)}"
    )
    print(
        f"Continuous price of the zero coupon bond in dollars: ${bond.calculate_continuous_price():.2f}"
    )

    bond = CouponBond(
        principal=1000,
        rate=10,
        maturity=3,
        interest_rate=4,
    )
    print(f"Price of the coupon bond in dollars: ${round(bond.calculate_price(), 2)}")
    print(
        f"Continuous price of the coupon bond in dollars: ${bond.calculate_continuous_price():.2f}"
    )
