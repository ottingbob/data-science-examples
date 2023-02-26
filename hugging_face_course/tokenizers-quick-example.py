from datasets import load_dataset
from transformers import AutoTokenizer

# If a tokenizer is not available in a language you are interested in or if the
# corpus is much different from the one your language model was trained on you
# most likely want to retrain the model from scratch using a tokenizer adapted
# to your data.

# For an example we will train GPT-2 from scratch using a corpus of Python Code
raw_datasets = load_dataset("code_search_net", "python")
print(raw_datasets["train"])

# In this dataset we will use the `whole_func_string` column to train our tokenizer.
print(raw_datasets["train"][123456]["whole_func_string"])


# We firstly need to transform the dataset into an iterator of lists of texts.
# This will enable our tokenizer to go faster and it should be an iterator to avoid
# having everything in memory at once.
# Here we create a generator to batch by lists of lists of 1_000 texts each:
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1_000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1_000)
    )


def get_training_corpus_yield():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1_000):
        samples = dataset[start_idx : start_idx + 1_000]
        yield samples["whole_func_string"]


old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
training_corpus = get_training_corpus_yield()
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)

# Now run the new tokenizer on our example
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''
tokens = tokenizer.tokenize(example)
print(tokens)
# ['def', 'Ġadd', '_', 'numbers', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`', 'a', '`', 'Ġand', 'Ġ`', 'b', '`."""', 'ĊĠĠĠ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb'])']

# Save the tokenizer
tokenizer.save_pretrained("code-search-net-tokenizer")
