from collections import defaultdict

import torch
from accelerated_models import AcceleratedGPT2
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    pipeline,
)

# We will build a scaled-down version of a code generation model.
# It focuses on one-line completions instead of full functions or classes by using
# a subset of Python code.


# We start by filtering the `codeparrot` dataset for all file that include any
# of the libraries in the data-science stack.
# Since the dataset is H U G E we avoid downloading it; we use the streaming
# feature to filter it on the fly.
def any_keyword_in_string(string, keywords):
    if True in [True for keyword in keywords if keyword in string]:
        return True
    else:
        return False


# Test on some examples
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
example_1 = "import numpy as np"
example_2 = "import pandas as pd"
print(
    any_keyword_in_string(example_1, filters),
    any_keyword_in_string(example_2, filters),
)


# Stream the dataset and filter the elements we want
def filter_streaming_dataset(dataset, filters):
    filtered_dict = defaultdict(list)
    total = 0
    for sample in tqdm(iter(dataset)):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content'])/total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)


# Now apply the function to the streaming dataset:
split = "train"
# This can take 2-3hrs so we grab the filtered dataset from the hub
# data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
# filtered_data = filter_streaming_dataset(data, filters)

ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
raw_datasets = DatasetDict(
    {
        "train": ds_train,
        "valid": ds_valid,
    }
)
print(raw_datasets)

# Look at an example from the dataset
for key in raw_datasets["train"][0]:
    print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")

# Now we need to tokenize the data to use for training. Since we want to autocomplete
# short function calls, we can keep the context size smaller. It also allows us to
# train the model much faster and requires significantly less memory.
# We use 128 tokens where GPT-2 uses 1024 and GPT-3 uses 2048
# We chunk the input instead of truncating them which would eliminate parts of our
# dataset. Then we also return the length of each chunk created, and if the last size
# is smaller than the chunk size, instead of padding we will just throw them out since
# we will have more than enough data.
context_length = 128
tokenizer = AutoTokenizer.from_pretrained(
    "huggingface-course/code-search-net-tokenizer"
)

# Here is an sample on the first two examples
outputs = tokenizer(
    raw_datasets["train"][:2]["content"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=True,
    return_length=True,
)
print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {outputs['length']}")
print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")


# With `Dataset.map()` we can create batches with more or fewer elements than the
# input batch. In our case tokenizing each element into chunks of the specified
# context size, we create many samples from each document. We make sure to delete
# the existing columns since they have a conflicting size
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}


# This will take about 35m to run...
tokenized_datasets = raw_datasets.map(
    tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
)
print(tokenized_datasets)
"""
We now have 16.7 million examples with 128 tokens each, which corresponds to about 2.1 billion tokens in total. For reference, OpenAI’s GPT-3 and Codex models are trained on 300 and 100 billion tokens, respectively, where the Codex models are initialized from the GPT-3 checkpoints
"""

# Now we initialize a GPT-2 model
config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
# We now load a new model, and we don't use a `from_pretrained()` function since
# we are initializing the model ourself
model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")

# We setup a DataCollator that can take care of creating batches. This will stack,
# batch, and create language model labels. In casual language modeling the inputs
# serve as labels too (just shifted by one element).
# The data collator creates them on the fly during training so we don't need
# to duplicate the `input_ids`
tokenizer.pad_token = tokenizer.eos_token
# We switch from masked language modeling to casual language modeling by setting
# the `mlm` argument to `False`
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
for key in out:
    print(f"{key} shape: {out[key].shape}")

# Now we configure the arguments for the Trainer
# We use a cosine learning rate schedule with warmup and an effective batch size
# of 256 (per_device_train_batch_size * gradient_accumulation_steps)
# Gradient accumulation is used when a single batch does not fit into memory,
# and incrementally builds up the gradient through several forward/backward passes.
# We see this in action when we create the training loop with the Accelerated model
args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    per_gpu_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)
# This can take up to 2 hours to run...
# It probably wouldn't work on my GPU anyways...
# trainer.train()

# Run an accelerated model -- won't work on my GPU =(
# am = AcceleratedGPT2(tokenized_datasets, tokenizer)
# am.execute(config)

# Now test the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe = pipeline(
    "text-generation",
    model="huggingface-course/codeparrot-ds",
    device=device,
)

# Start with a test to create a scatter plot
txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

# Test to see if we can create a DataFrame from two arrays:
txt = """\
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create dataframe from x and y
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

# Test to see if the model can use a groupby operation
txt = """\
# dataframe with profression, income and name
df = pd.DataFrame({'profession': x, 'income': y, 'name': z})

# calculate the mean income per profession
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])

# Test to see if it can use scikit-learn to setup a Random Forest model
txt = """\
# import random forest regressor from scikit-learn
from sklearn.ensemble import RandomForestRegressor

# fit random forest model with 300 estimators on X, y:
"""
print(pipe(txt, num_return_sequences=1)[0]["generated_text"])
