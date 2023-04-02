import re
from contextlib import contextmanager
from typing import Any, Dict, Tuple


class TermColors:
    HEADER = "\033[95m"
    OKBLUE = "\033[34m"
    OKYELLOW = "\033[33m"
    OKWHITE = "\033[37m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[32m"
    WARNING = "\033[93m"
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
        header_color = (f"{cls.OKBLUE}{cls.BOLD}%s{cls.ENDC}",)
        positive_color = (f"{cls.OKGREEN}{cls.BOLD}%s{cls.ENDC}",)
        yellow_color = (f"{cls.OKYELLOW}{cls.BOLD}%s{cls.ENDC}",)
        negative_color = (f"{cls.FAIL}{cls.BOLD}%s{cls.ENDC}",)
        # any_value_pattern = re.compile(r"┆\s+(\w+)\s+│")
        any_value_pattern = re.compile(r"(?=([┆│]\s+([\w\-\s%]+)\s+[┆│]))")
        number_pattern = re.compile(r"(\-?[0-9]+\.[0-9]+%?)")
        year_pattern = re.compile(r"([0-9]{4})")

        def compile_label_regex(row_label: str) -> re.Pattern:
            return re.compile(r"^\\|\s+?(%s)" % row_label)

        # Create a label regex color map
        def create_lrcm(
            row_regex: re.Pattern, row_color: Tuple[str], value_pattern: re.Pattern
        ) -> Dict[str, Any]:
            return {
                "row_regex": row_regex,
                "row_color": row_color[0],
                "value_pattern": value_pattern,
            }

        def positive_number_lrcm(row_label: str):
            return create_lrcm(
                row_regex=compile_label_regex(row_label),
                row_color=positive_color,
                value_pattern=number_pattern,
            )

        def negative_number_lrcm(row_label: str):
            return create_lrcm(
                row_regex=compile_label_regex(row_label),
                row_color=negative_color,
                value_pattern=number_pattern,
            )

        label_regex_color_map = [
            create_lrcm(
                compile_label_regex("Mapping"),
                header_color,
                any_value_pattern,
            ),
            positive_number_lrcm("Revenue"),
            positive_number_lrcm("Gross Profit"),
            positive_number_lrcm("EBITDA"),
            positive_number_lrcm("EBIT"),
            positive_number_lrcm("EBT"),
            positive_number_lrcm("Net Income"),
            negative_number_lrcm("Cogs"),
            negative_number_lrcm("Operating expenses"),
            negative_number_lrcm("Interest expenses"),
            negative_number_lrcm("D&A"),
            negative_number_lrcm("Taxes"),
        ]

        def color_line(
            line_text: str,
            row_label: str,
            color_format: str,
            values_regex: re.Pattern = number_pattern,
        ) -> str:
            line_text = line_text.replace(row_label, color_format % (row_label))
            for value in values_regex.findall(line_text):
                # If it is a header value we need to get the last group value
                if "Mapping" in row_label:
                    line_text = line_text.replace(value[-1], color_format % (value[-1]))
                # If it is a percentage we need to format the color conditionally
                elif "%" in value:
                    value_no_percent = float(value[: value.index("%")])
                    if value_no_percent > 10.0:
                        line_text = line_text.replace(
                            value, positive_color[0] % (value)
                        )
                    elif value_no_percent > 0:
                        line_text = line_text.replace(value, yellow_color[0] % (value))
                    else:
                        line_text = line_text.replace(
                            value, negative_color[0] % (value)
                        )
                # Otherwise the formatting is straight forward on a single value group match
                else:
                    # FIXME: Having an issue where this can clobber coloring if the value has been
                    # found or is a substring of another earlier value
                    # Only replace last occurrence
                    # TODO: Unsure which is faster..?
                    # return (s[::-1].replace(old[::-1], new[::-1], count))[::-1]
                    # line_text = (color_format % (value)).join(line_text.rsplit(value, 1))
                    line_text = line_text.replace(value, color_format % (value))
            return line_text

        # Apply regex patterns over lines
        # We create an iterator to iterate more than 1 line when we want to skip 2
        text_idx_iter = iter(enumerate(text.splitlines()))
        for idx, line_text in text_idx_iter:
            # exclude shape row
            if line_text.startswith("shape:"):
                continue
            elif line_text.startswith("│ ---"):
                next(text_idx_iter)
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
