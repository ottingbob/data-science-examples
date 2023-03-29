import re
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, Tuple


class TermColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[34m"
    OKWHITE = "\033[37m"
    OKCYAN = "\033[96m"
    # OKGREEN = "\033[92m"
    OKGREEN = "\033[32m"
    WARNING = "\033[93m"
    # FAIL = "\033[91m"
    FAIL = "\033[31m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    @classmethod
    def df_header(cls, header_text: str) -> str:
        return f"\n{cls.OKGREEN}{cls.BOLD}{header_text}{cls.ENDC}\n"

    @classmethod
    def print_pandl_with_colors(cls, text: str):
        # Compile regex patterns
        header_row = re.compile(r"^\\|\s+?(Mapping)")
        header_color = (f"{cls.OKBLUE}{cls.BOLD}%s{cls.ENDC}",)
        profit_row = re.compile(r"^\\|\s+?(Gross Profit)")
        positive_color = (f"{cls.OKGREEN}{cls.BOLD}%s{cls.ENDC}",)
        cogs_row = re.compile(r"^\\|\s+?(Cogs)")
        negative_color = (f"{cls.FAIL}{cls.BOLD}%s{cls.ENDC}",)
        rev_row = re.compile(r"^\\|\s+?(Revenue)")
        opex_row = re.compile(r"^\\|\s+?(Operating expenses)")
        d_and_a_row = re.compile(r"^\\|\s+?(D&A)")
        ebitda = re.compile(r"^\\|\s+?(EBITDA)")
        ebit = re.compile(r"^\\|\s+?(EBIT)")
        ebt = re.compile(r"^\\|\s+?(EBT)")
        net_income_row = re.compile(r"^\\|\s+?(Net Income)")
        taxes_row = re.compile(r"^\\|\s+?(Taxes)")
        intex_row = re.compile(r"^\\|\s+?(Interest expenses)")
        number_pattern = re.compile(r"([0-9]+\.[0-9]+)")
        year_pattern = re.compile(r"([0-9]{4})")

        def create_lrcm(
            row_regex: re.Pattern, row_color: Tuple[str], value_pattern: re.Pattern
        ) -> Dict[str, Any]:
            return {
                "row_regex": row_regex,
                "row_color": row_color[0],
                "value_pattern": value_pattern,
            }

        positive_number_lrcm = partial(
            create_lrcm, row_color=positive_color, value_pattern=number_pattern
        )
        negative_number_lrcm = partial(
            create_lrcm, row_color=negative_color, value_pattern=number_pattern
        )

        label_regex_color_map = [
            create_lrcm(header_row, header_color, year_pattern),
            positive_number_lrcm(rev_row),
            positive_number_lrcm(profit_row),
            positive_number_lrcm(ebitda),
            positive_number_lrcm(ebit),
            positive_number_lrcm(ebt),
            positive_number_lrcm(net_income_row),
            negative_number_lrcm(cogs_row),
            negative_number_lrcm(opex_row),
            negative_number_lrcm(intex_row),
            negative_number_lrcm(d_and_a_row),
            negative_number_lrcm(taxes_row),
        ]

        def color_line(
            line_text: str,
            row_label: str,
            color_format: str,
            values_regex: re.Pattern = number_pattern,
        ) -> str:
            line_text = line_text.replace(row_label, color_format % (row_label))
            for value in values_regex.findall(line_text):
                line_text = line_text.replace(value, color_format % (value))
            return line_text

        # Apply regex patterns over lines
        for idx, line_text in enumerate(text.splitlines()):
            # exclude shape row
            if line_text.startswith("shape:"):
                continue

            for lrcm in label_regex_color_map:
                if label := lrcm["row_regex"].search(line_text):
                    line_text = color_line(
                        line_text,
                        label.group(),
                        lrcm["row_color"],
                        lrcm["value_pattern"],
                    )
                    break
            """
            if header_label := header_row.search(line_text):
                line_text = color_line(
                    line_text,
                    header_label.group(),
                    f"{cls.OKBLUE}{cls.BOLD}%s{cls.ENDC}",
                    year_pattern,
                )
            elif rev_label := rev_row.search(line_text):
                line_text = color_line(
                    line_text,
                    rev_label.group(),
                    f"{cls.OKWHITE}{cls.BOLD}%s{cls.ENDC}",
                )
            elif cogs_label := cogs_row.search(line_text):
                line_text = color_line(
                    line_text,
                    cogs_label.group(),
                    f"{cls.FAIL}{cls.BOLD}%s{cls.ENDC}",
                )
            elif profit_label := profit_row.search(line_text):
                line_text = color_line(
                    line_text,
                    profit_label.group(),
                    f"{cls.OKGREEN}{cls.BOLD}%s{cls.ENDC}",
                )
            """
            print(line_text)

    @classmethod
    def _log_failure(cls, e: Exception) -> None:
        print(f"{cls.FAIL}{cls.BOLD}{e}{cls.ENDC}")

    @classmethod
    @contextmanager
    def with_failures(cls):
        try:
            yield
        except Exception as e:
            cls._log_failure(e)
