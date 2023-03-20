import xml.etree.ElementTree as ElT
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup


def parse_xml_to_csv(path: Path) -> pd.DataFrame:
    """
    Open .xml posts dump and convert the text to a csv, tokenizing it in the process
    :param path: path to the xml document containing posts
    :return: a dataframe of processed text
    """

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
