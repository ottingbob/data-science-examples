import timeit

import psutil
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
"""
pubmed_dataset = load_dataset(
    "json",
    # "default-6e3092816c4f845b",
    # data_dir="default-6e3092816c4f845b",
    data_files=data_files,
    split="train",
)
pubmed_dataset.save_to_disk("pubmed_dataset")
"""

pubmed_dataset = load_from_disk("pubmed_dataset")
print(pubmed_dataset)

# Show the contents of the first example:
print(pubmed_dataset[1])

# Show how much memory usage of our current process since we are working with 15M records
# `memory_info()` is expressed in bytes, so convert it to megabytes
# `rss` means resident set size, which is the fraction of memory that a process occupies
# in RAM. This includes memory from the python interpreter, loaded libs, etc.
memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)
print(f"RAM used: {memory_usage:.2f} MB")

# Print the dataset size on disk
print(f"Number of files in dataset: {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size / (1024**3)
print(f"Dataset size (cache file): {size_gb:.2f} GB")

# For memory management the datasets library treats each dataset as a memory-mapped
# file which provides a mapping between RAM and filesystem storage that allows for
# the library to conditionally access and operate on elements of the dataset without
# needing to fully load it into memory.

code_snippit = """
batch_size = 1000

for idx in range(0, len(pubmed_dataset), batch_size):
    _ = pubmed_dataset[idx:idx + batch_size]
"""
"""
time = timeit.timeit(stmt=code_snippit, number=1, globals=globals())
print(
    f"Iterated over {len(pubmed_dataset)} examples (about {size_gb:.1f} GB) in "
    f"{time:.1f}s, i.e. {size_gb/time:.3f} GB/s"
)
"""

# We can enable streaming of the dataset to iterate faster. This will return an
# `IterableDataset`
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)
print(next(iter(pubmed_dataset_streamed)))

# The elements from a streamed dataset can be processed using the `map()` function
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = pubmed_dataset_streamed.map(
    lambda x: tokenizer(x["text"], batched=True)
)
print(next(iter(tokenized_dataset)))
