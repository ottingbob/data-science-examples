import collections
import math

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    default_data_collator,
    pipeline,
)

# DistilBERT is trained from BERT and has far fewer parameters
# and pretrained on English Wikipedia and BookCorpus datasets
model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

text = "This is a great [MASK]."

# We need DistilBERTs tokenizer to produce the inputs for the model
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Pass our text example to the model, extract the logits, and print
# out the top 5 candidates
inputs = tokenizer(text, return_tensors="pt")
token_logits = model(**inputs).logits
# Find the location of the [MASK] and extract its logits
mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
# Pick the [MASK] candidates with the highest logits
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")

# Now we change the model to be tailored towards movie reviews
# We will use the IMDb corpus of movie reviews which is often times used to benchmark
# sentiment analysis models
imdb_dataset = load_dataset("imdb")
print(imdb_dataset)

# Check out an example of some of the data
sample = imdb_dataset["train"].shuffle(seed=42).select(range(3))
for row in sample:
    print(f"\n>>> Review: {row['text']}")
    print(f"\n>>> Label: {row['label']}")


# Prepare the data for masked langage modeling
# For auto-regressive & masked language modeling, a common preprocessing step is to
# concatenate all the examples and then split the whole corpus into chunks of equal size.
# The reason we concatenate instead of tokenizing individual examples is they might get
# truncated if they are too long, and we would lose information useful for the modeling task
def tokenize_function(examples):
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i) for i in range(len(result["input_ids"]))
        ]
    return result


tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
print(tokenized_datasets)

chunk_size = 128
# Here is how the concatenation works:
# Slicing produces a list of lists for each feature
tokenized_samples = tokenized_datasets["train"][:3]
for idx, sample in enumerate(tokenized_samples["input_ids"]):
    print(f">>> Review {idx} length: {len(sample)}")

# Concatenate all these examples with a simple dict comprehension
concatenated_examples = {
    k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
}
total_length = len(concatenated_examples["input_ids"])
print(f">>> Concatenated reviews length: {total_length}")

# Split the concatenated reviews into the chunks of the size we defined
chunks = {
    k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
    for k, t in concatenated_examples.items()
}
for chunk in chunks["input_ids"]:
    print(f">>> Chunk length: {len(chunk)}")


# With the last chunk being < chunk_size we can drop the first chunk when
# we create a function to apply it to our tokenized datasets
def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in tokenized_samples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Drop the last chunk if it's smaller than chunk size
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column -- we do this because in masked language
    # modeling we predict randomly masked tokens in the input batch, and by
    # creating a labels column we provide the ground truth for our language
    # model to learn from
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True)
# Grouping & chunking the texts produces more examples than the original 25_000 for the
# train and test splits. Since we have examples involving contiguous tokens that span
# across multiple examples from the original corpus we have this behavior
print(lm_datasets)

# Now for training purposes we need to insert the [MASK] token at random positions on
# the inputs using a special data collator
# We choose an mlm_probability of 15% which is the amount used for BERT and a common
# choice in other literature
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm_probability=0.15
)

# Look at an example of how random masking works
samples = [lm_datasets["train"][i] for i in range(2)]
for sample in samples:
    _ = sample.pop("word_ids")
for chunk in data_collator(samples)["input_ids"]:
    print(f"\n>>> {tokenizer.decode(chunk)}")

# Whole word masking masks whole words together instead of just individual tokens.
# If we want to do this we need to build a data collator ourselves
# Labels are all `-100` except for the ones corresponding to mask words
wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
        feature["labels"] = new_labels

    return default_data_collator(features)


# Try it on the same samples as before
samples = [lm_datasets["train"][i] for i in range(2)]
batch = whole_word_masking_data_collator(samples)
for chunk in batch["input_ids"]:
    print(f"\n>>> {tokenizer.decode(chunk)}")

# Tune the training size down to speed up but still produce a decent model
train_size = 10_000
test_size = int(0.1 * train_size)
downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
print(downsampled_dataset)

# Now setup the trainer
# batch_size = 64
# 2% || 8/471 [01:44<1:41:53, 13.20s/it]

# batch_size = 16
import os

batch_size = os.cpu_count()
# 2% || 43/2502 [01:36<1:54:36,  2.80s/it]

# batch_size = 32
# 0% || 4/939 [00:21<1:19:20,  5.09s/it]

logging_steps = len(downsampled_dataset["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]
training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-imdb",
    # overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=False,
    # Mixed precision training to get a boost in speed
    # Can only be used on GPU training
    # fp16=True,
    # Track the training loss with each epoch
    logging_steps=logging_steps,
    # dont use GPU and run out of memory
    no_cuda=True,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# We can now use `perplexity` which is a common metric to evaluate the
# performance of language models.
# We use cross-entropy loss to calculate whether our model is good at
# assigining the next word in the sentences of our test set. High probabilities
# will indicate the model is not `perplexed` by the unseen examples and suggest
# it has learned the basic patterns of grammar in the language
#
# eval_results = trainer.evaluate()
# print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
# >>> Perplexity: 21.94

# trainer.train()

# eval_results = trainer.evaluate()
# print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# And now use the fine-tuned model
# I am not waiting an hour+ for training so using the model from the course instead
# mask_filler = pipeline("fill-mask", model=f"{model_name}-finetuned-imdb")
mask_filler = pipeline(
    "fill-mask", model="huggingface-course/distilbert-base-uncased-finetuned-imdb"
)
preds = mask_filler(text)
for pred in preds:
    print(f">>> {pred['sequence']}")

# TODO: We can use a custom training loop... IF YOU DARE
