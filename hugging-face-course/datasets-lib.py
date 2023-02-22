import html
from pathlib import Path
from zipfile import ZipFile

import requests
from datasets import load_dataset


# Load a remote dataset
def load_remote_dataset():
    url = "https://github.com/crux82/squad-it/raw/master/"
    data_files = {
        "train": f"{url}SQuAD_it-train.json.gz",
        "test": f"{url}SQuAD_it-test.json.gz",
    }
    squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
    print(squad_it_dataset)


dataset_file = Path("hugging-face-course/drugsCom_raw.zip")
if not dataset_file.exists():
    dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00462/drugsCom_raw.zip"
    resp = requests.get(dataset_url)
    with dataset_file.open("wb") as f:
        f.write(resp.content)
    # Unzip the files
    with ZipFile(dataset_file.resolve(), "r") as z:
        z.extractall("hugging-face-course/")

data_files = {
    "train": "hugging-face-course/drugsComTrain_raw.tsv",
    "test": "hugging-face-course/drugsComTest_raw.tsv",
}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
print(drug_dataset)

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
print(drug_sample[:3])

# Verify number of unique patient IDs
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

# Rename the ID column
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset)


# Normalize the condition labels
def lowercase_condition(drug_dataset_row):
    return {"condition": drug_dataset_row["condition"].lower()}


# Remove `None` conditions using a lambda / filter function
drug_dataset = drug_dataset.filter(lambda row: row["condition"] is not None)
drug_dataset = drug_dataset.map(lowercase_condition)
print(drug_dataset["train"]["condition"][:3])


# Add review length to the dataset
# This could also be achieved with `dataset.add_column()` if a map does not
# meet the needs of what data needs to be added.
def compute_review_length(drug_dataset_row):
    return {"review_length": len(drug_dataset_row["review"].split())}


drug_dataset = drug_dataset.map(compute_review_length)
print(drug_dataset["train"][0])

# Use filter to remove reviews that container less than 30 words
drug_dataset = drug_dataset.filter(lambda row: row["review_length"] > 30)
print(drug_dataset.num_rows)

# Unescape HTML character codes from the reviews
drug_dataset = drug_dataset.map(
    lambda row: {"review": html.unescape(row["review"])}, batched=True
)
