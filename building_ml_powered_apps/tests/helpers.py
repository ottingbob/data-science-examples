import xml.etree.ElementTree as ElT
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


def parse_xml_to_csv(curr_path: Path) -> pd.DataFrame:
    """
    Open .xml posts dump and convert the text to a csv, tokenizing it in the process
    :param curr_path: path to the current testing file
    :return: a dataframe of processed text
    """
    # Find parent directory for `data-science-examples`
    base_path = ""
    for p in curr_path.parents:
        if str(p).split("/")[-1] == "data-science-examples":
            base_path = p
    path = Path(
        str(base_path) + "/ml-powered-applications/tests/fixtures/MiniPosts.xml"
    )

    # Use python's standard library to parse XML file
    doc = ElT.parse(path)
    root = doc.getroot()

    # Each row is a question
    all_rows = [row.attrib for row in root.findall("row")]

    for item in all_rows:
        # Decode text from HTML
        soup = BeautifulSoup(item["Body"], features="html.parser")
        item["body_text"] = soup.get_text()

    # Create dataframe from our list of dictionaries
    return pd.DataFrame.from_dict(all_rows)
