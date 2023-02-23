import html
import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer


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

# We can also increase the number of processes by providing a `num_proc`
# arg to the map methods
cpus = os.cpu_count()

# Unescape HTML character codes from the reviews
drug_dataset = drug_dataset.map(
    lambda row: {"review": html.unescape(row["review"])}, batched=True, num_proc=cpus
)

# In ML an `example` is dfined as the set of features that we feed into a model.
# These can either be the set of columns in a dataset or multiple features can be
# extracted from a single example and belong to a single column
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")


def tokenize_and_split(examples):
    # We can deal with mismatched length by making the old columns the same size as
    # the new ones. We need the `overflow_to_sample_mapping` field returned here.
    # It will map the new feature index to the one of the sample it originated from.
    # We can then associate each key in the original dataset with a list of values
    # of the right size by repeating values of each sample when it generates new features.
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # Extract mapping between new and old indices
    sample_map = result.pop("overflow_to_sample_mapping")
    for k, v in examples.items():
        result[k] = [v[i] for i in sample_map]
    return result


# Here our first example in the training set becomes 2 features because it will
# tokenize to more than the max number of tokens we specified, `128`:
# result = tokenize_and_split(drug_dataset["train"][0])
# print([len(inp) for inp in result["input_ids"]])
# [128, 49]

# We remove columns from the drug dataset to not go over our column length of `1000`
"""
tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
"""
# Instead use the updated `tokenize_and_split` method that maps the new features back to
# the original ones.
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
print(len(tokenized_dataset["train"]), len(drug_dataset["train"]))

# Convert the dataset to a dataframe object:
drug_dataset.set_format("pandas")
print(drug_dataset["train"][:3])

# Create a DataFrame for the whole training set
train_df: pd.DataFrame = drug_dataset["train"][:]
# Use pandas to compute the class distribution among the `condition` entries:
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "condition": "frequency"})
)
print(frequencies.head())

# Now we can go back to a dataset object:
freq_dataset = Dataset.from_pandas(frequencies)
print(freq_dataset)

# Reset the `drug_dataset` from `pandas` to `arrow` for further processing
drug_dataset.reset_format()

# Split the training set into train and validation splits:
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# Default "test" split to "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# Add the "test" set to our `DatasetDict`
drug_dataset_clean["test"] = drug_dataset["test"]
print(drug_dataset_clean)

# Save the dataset to disk
# drug_dataset_clean.save_to_disk("drug-reviews")
drug_dataset_reloaded = load_from_disk("drug-reviews")
print(drug_dataset_reloaded)
