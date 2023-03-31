# Accounting transactions on the Lemonade Stand
from datetime import datetime
from enum import Enum, auto
from typing import Dict

import polars as pl


class BalanceSheetType(Enum):
    ASSET = auto()
    LIABILITY_AND_EQUITY = auto()


class IncomeStatementType(Enum):
    EXPENSES = auto()
    INCOME = auto()


class BalanceSheet:
    BALANCE_SHEET_CATEGORIES: Dict[str, BalanceSheetType] = {
        "Cash": BalanceSheetType.ASSET,
        "PP&E": BalanceSheetType.ASSET,
        "Raw Materials": BalanceSheetType.ASSET,
        "Equity": BalanceSheetType.LIABILITY_AND_EQUITY,
        "Trade Payables": BalanceSheetType.LIABILITY_AND_EQUITY,
    }

    INCOME_STATEMENT_CATEGORIES: Dict[str, IncomeStatementType] = {
        "Cogs": IncomeStatementType.EXPENSES,
        "Utility Expenses": IncomeStatementType.EXPENSES,
        "D&A": IncomeStatementType.EXPENSES,
        "Revenue": IncomeStatementType.INCOME,
    }

    def __init__(self):
        self.balance_sheet = pl.DataFrame(
            {
                "Date": [],
                "Account": [],
                "Debit": [],
                "Credit": [],
            },
            schema={
                "Date": pl.Date,
                "Account": pl.Utf8,
                "Debit": pl.Float64,
                "Credit": pl.Float64,
            },
        )

    def __str__(self):
        return str(self.balance_sheet)

    def add_balance_sheet_row(
        self,
        date: datetime,
        account: str = "",
        debit: float = 0.0,
        credit: float = 0.0,
    ) -> pl.DataFrame:
        self.balance_sheet = self.balance_sheet.vstack(
            pl.DataFrame(
                {
                    "Date": date,
                    "Account": account,
                    "Debit": float(debit),
                    "Credit": float(credit),
                }
            )
        )

    def calculate_balance(self, is_income_statement: bool = False) -> float:
        if is_income_statement:
            return self._calculate_income()
        else:
            return self._calculate_balance()

    def _calculate_balance(self):
        balance_sheet = (
            self.balance_sheet.groupby("Account")
            .agg([pl.col("Debit").sum(), pl.col("Credit").sum()])
            .with_columns(
                [
                    pl.struct([pl.col("Account"), pl.col("Debit"), pl.col("Credit")])
                    .apply(
                        lambda x: x["Debit"] - x["Credit"]
                        if BalanceSheet.BALANCE_SHEET_CATEGORIES[x["Account"]]
                        == BalanceSheetType.ASSET
                        else x["Credit"] - x["Debit"]
                    )
                    .alias("Total"),
                    pl.col("Account")
                    .apply(lambda x: BalanceSheet.BALANCE_SHEET_CATEGORIES[x].name)
                    .alias("Category"),
                ]
            )
            .groupby(pl.col("Category"))
            .agg(pl.col("Total").sum())
        )
        print(balance_sheet)
        # Remaining balance should be net income
        remaining_balance = (
            balance_sheet.filter(pl.col("Category") == BalanceSheetType.ASSET.name)
            .select("Total")
            .item()
            - balance_sheet.filter(
                pl.col("Category") == BalanceSheetType.LIABILITY_AND_EQUITY.name
            )
            .select("Total")
            .item()
        )
        return remaining_balance

    def _calculate_income(self) -> float:
        income_statement = (
            self.balance_sheet.groupby("Account")
            .agg([pl.col("Debit").sum(), pl.col("Credit").sum()])
            .with_columns(
                [
                    pl.struct([pl.col("Account"), pl.col("Debit"), pl.col("Credit")])
                    .apply(
                        lambda x: x["Debit"] - x["Credit"]
                        if BalanceSheet.INCOME_STATEMENT_CATEGORIES[x["Account"]]
                        == IncomeStatementType.EXPENSES
                        else x["Credit"] - x["Debit"]
                    )
                    .alias("Total"),
                    pl.col("Account")
                    .apply(lambda x: BalanceSheet.INCOME_STATEMENT_CATEGORIES[x].name)
                    .alias("Category"),
                ]
            )
            # .with_columns([pl.col("Total"), pl.col("Category")])
            .groupby(pl.col("Category"))
            .agg(pl.col("Total").sum())
        )
        print(income_statement)
        net_income = (
            income_statement.filter(
                pl.col("Category") == IncomeStatementType.INCOME.name
            )
            .select("Total")
            .item()
            - income_statement.filter(
                pl.col("Category") == IncomeStatementType.EXPENSES.name
            )
            .select("Total")
            .item()
        )
        return net_income


balance_sheet = BalanceSheet()
# Firm received 3_000 so make the following adjustments
#
# In polars we can use `vstack` to add rows cheaply but after a while should do a `pl.concat`
# if we need to rechunk and do a new memory slab
balance_sheet.add_balance_sheet_row(datetime(2015, 6, 1).date(), "Equity", 0, 3_000)
balance_sheet.add_balance_sheet_row(datetime(2015, 6, 1).date(), "Cash", 3_000, 0)

# Firm acquires lemonade stand for 1_000 (PP&E = Property Plant & Equipment)
# since this is an asset we debit the account
balance_sheet.add_balance_sheet_row(datetime(2015, 6, 10).date(), "PP&E", 1_000, 0)
balance_sheet.add_balance_sheet_row(datetime(2015, 6, 10).date(), "Cash", 0, 1_000)

# Firm purchases raw materials for 500 and pay 250 now and 250 in 2 weeks
balance_sheet.add_balance_sheet_row(
    datetime(2015, 6, 12).date(), "Raw Materials", 500, 0
)
balance_sheet.add_balance_sheet_row(datetime(2015, 6, 12).date(), "Cash", 0, 250)
balance_sheet.add_balance_sheet_row(
    datetime(2015, 6, 12).date(), "Trade Payables", 0, 250
)

# Part 2
# In the next 2 weeks the firm:
# - Spends 400 in raw materials
# - Generates 1200 in revenue
# - Pays 150 in utility bills

# FIXME: This could have a more generic naming convention...
income_statement = BalanceSheet()
balance_sheet.add_balance_sheet_row(
    datetime(2015, 6, 26).date(), "Raw Materials", 0, 400
)
income_statement.add_balance_sheet_row(datetime(2015, 6, 26).date(), "Cogs", 400, 0)

balance_sheet.add_balance_sheet_row(datetime(2015, 6, 26).date(), "Cash", 1200, 0)
income_statement.add_balance_sheet_row(datetime(2015, 6, 26).date(), "Revenue", 0, 1200)

balance_sheet.add_balance_sheet_row(datetime(2015, 6, 12).date(), "Cash", 0, 150)
income_statement.add_balance_sheet_row(
    datetime(2015, 6, 12).date(), "Utility Expenses", 150, 0
)

# Part 3
# July 15 -- purchases 900 of raw materials
# August 20 -- sells 2700 for 900 cogs
# August 20 -- utility bill for 200
# August 20 -- Accrews 100 of depreciation
balance_sheet.add_balance_sheet_row(datetime(2015, 7, 15).date(), "Cash", 0, 900)
balance_sheet.add_balance_sheet_row(
    datetime(2015, 7, 15).date(), "Raw Materials", 900, 0
)

balance_sheet.add_balance_sheet_row(
    datetime(2015, 8, 20).date(), "Raw Materials", 0, 900
)
income_statement.add_balance_sheet_row(datetime(2015, 8, 20).date(), "Cogs", 900, 0)

balance_sheet.add_balance_sheet_row(datetime(2015, 8, 20).date(), "Cash", 2700, 0)
income_statement.add_balance_sheet_row(datetime(2015, 8, 20).date(), "Revenue", 0, 2700)

balance_sheet.add_balance_sheet_row(datetime(2015, 8, 20).date(), "Cash", 0, 200)
income_statement.add_balance_sheet_row(
    datetime(2015, 8, 20).date(), "Utility Expenses", 200, 0
)

balance_sheet.add_balance_sheet_row(datetime(2015, 8, 20).date(), "PP&E", 0, 100)
income_statement.add_balance_sheet_row(datetime(2015, 8, 20).date(), "D&A", 100, 0)

# Part 4
# Calculate totals for each category
# We also need to calculate `Net Income` in order to balance the balance sheet
remaining_balance = balance_sheet.calculate_balance()
net_income = income_statement.calculate_balance(is_income_statement=True)

# TODO: We could account for net_income as company equity
# TODO: Write a method to print the income statement with rows we would expect
assert remaining_balance == net_income
print("Net Income:", net_income)
