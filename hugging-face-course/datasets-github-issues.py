import math
import os
import time
from pathlib import Path

import pandas as pd
import requests
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Create a corpus of GitHub issues which can be used for purposes such as:
# - exploring how long it takes to close open issues or pull requests
# - training a multilabel classifier that can tag issues with metadata
#   based on the issues description (bug, enhancement, question)
# - creating a semantic search engine to find which issues match a users
#   query

# Getting the data
# url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
# response = requests.get(url)

load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
headers = {"Authorization": f"token {GITHUB_TOKEN}"}


def fetch_issues(
    owner="huggingface",
    repo="datasets",
    num_issues=10_000,
    rate_limit=5_000,
    issues_path: Path = Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(exist_ok=True)

    batch = []
    all_issues = []
    per_page = 100
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    # TODO: Would need to reset on a new page
    progress_bar = tqdm(range(num_pages))
    for page in range(num_pages):
        # Query with state=all to get both open and closed issues
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        batch.extend(issues.json())
        progress_bar.update(1)

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            batch = []
            print("Reached GitHub rate limit. Stopping execution...")
            # print(f"Reached GitHub rate limit. Sleeping for one hour...")
            # time.sleep(60 * 60 + 1)
            break

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(
        f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl"
    )


# Get _all_ the data
# fetch_issues()
# TODO: Fix parsing the dataset...?
issues_df = pd.read_json("datasets-issues.jsonl", lines=True)
# issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")
issues_dataset = Dataset.from_pandas(issues_df)
print(issues_dataset)


# Clean up the data
def print_sample(ds: Dataset):
    sample = ds.shuffle(seed=666).select(range(3))
    for url, pr, is_pr in zip(
        sample["html_url"], sample["pull_request"], sample["is_pull_request"]
    ):
        print(f">> URL: {url}")
        print(f">> Pull request: {pr}")
        print(f">> IS Pull request: {is_pr}\n")


# Create an `is_pull_request` column that checks if the issue is a pullrequest
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True},
    # batched=True,
)
print_sample(issues_dataset)

# Can additionally do things like filter out pull requests / open issues,
# convert dataset to dataframe to manipulate created_at / closed_at timestamps
# have a calculation for average time to close pull requests etc.


# Add comments to the dataset
def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]


issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)
issues_with_comments_dataset.save_to_disk("datasets-issues-with-comments")
# df.to_json("/datasets-issues-with-comments.jsonl", orient="records", lines=True)
