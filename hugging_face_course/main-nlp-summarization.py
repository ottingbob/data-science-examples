import evaluate
import nltk
import numpy as np
import pandas as pd
from accelerated_models import AcceleratedMT5
from datasets import DatasetDict, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    pipeline,
)

# We use a multilingual amazon review corpus to create a bilingual summarizer
# In this example we use the short titles as target summaries for our model
# to learn from.
spanish_dataset = load_dataset("amazon_reviews_multi", "es")
english_dataset = load_dataset("amazon_reviews_multi", "en")
print(english_dataset)

# Here we show some samples from the dataset
sample = english_dataset["train"].shuffle(seed=42).select(range(3))
for example in sample:
    print(f"\n>> Title: {example['review_title']}")
    print(f"\n>> Review: {example['review_body']}")

# We focus on generating summaries for a single domain of products
# instead of looking at all 400k reviews (200 per language)
# Show the number of reviews per product category
english_dataset.set_format("pandas")
english_df = english_dataset["train"][:]
# Show counts for top 20 products
print(english_df["product_category"].value_counts()[:20])


# We choose to stick with the `book` and `digital_ebook_purchase` categories
def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )


# Switch from pandas back to arrow
english_dataset.reset_format()
# Apply the filter function and inspect a sample of the reviews
spanish_books = spanish_dataset.filter(filter_books)
english_books = english_dataset.filter(filter_books)
sample = english_books["train"].shuffle(seed=42).select(range(3))
for example in sample:
    print(f"\n>> Title: {example['review_title']}")
    print(f"\n>> Review: {example['review_body']}")

# Combine the English and Spanish reviews as a single `DatasetDict`
books_dataset = DatasetDict()
for split in english_books.keys():
    books_dataset[split] = concatenate_datasets(
        [english_books[split], spanish_books[split]]
    )
    books_dataset[split] = books_dataset[split].shuffle(seed=42)

# And show some samples now
sample = books_dataset["train"].shuffle(seed=42).select(range(3))
for example in sample:
    print(f"\n>> Title: {example['review_title']}")
    print(f"\n>> Review: {example['review_body']}")

# The course gets an idea of the distribution of # of words per title
# and the review body.
book_df = pd.DataFrame.from_dict(english_books.copy()["train"])
print(book_df.head())
# title_counts = {}
title_counts = book_df["review_title"].str.split().apply(len).value_counts()
# title_counts.index = title_counts.index.astype(str) + " words:"
title_counts.index = title_counts.index.astype(int)
title_counts.sort_index(inplace=True)
print(title_counts)

# We filter out titles < 3 words to get more interesting summaries
books_dataset = books_dataset.filter(lambda x: len(x["review_title"].split()) > 2)

# Now we tokenize and encode our reviews and their titles. We load a tokenizer
# with our pretrained model checkpoint, which in this case is `mt5-small`
model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
# And test it out on a small sample
inputs = tokenizer("I loved reading the Hunger Games!")
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs.input_ids))

# To tokenize with summarization we need to deal with labels that are also
# text, so it is possible they exceed the models maximum context size
# We need to apply truncation to both the reviews and their titles to ensure
# we don't pass too long of inputs to the model
max_input_length = 512
max_target_length = 30


def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
    )
    labels = tokenizer(
        examples["review_title"], max_length=max_target_length, truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_datasets = books_dataset.map(preprocess_function, batched=True)
# For a metric we use the `ROUGE` score (Recall-Oriented Understudy for Gisting
# Evaluation). This uses precision and recall scores for overlaps between comparisons
# Recall measures: (# of overlapping words) / (total # of words in reference summary)
# Precision measures (# of overlapping words) / (total # of words in generated summary)
# We install the `rouge_score` package to use it as a metric
rouge_score = evaluate.load("rouge")

# Get test metrics
generated_summary = "I absolutely loved reading the Hunger Games"
reference_summary = "I loved reading the Hunger Games"
scores = rouge_score.compute(
    predictions=[generated_summary], references=[reference_summary]
)
print(scores)

# We create a baseline for text summarization by taking the first 3 sentences of
# an article, called the lead-3 baseline. We use full stops to track the sentence
# boundaries and use `nltk` to handle cases for acronyms such as `U.S.` or `U.N.`
# Here we get the punctuation rules:
nltk.download("punkt")


# Now import the sentence tokenizer and create a simple function to get the first
# 3 sentences of a review. The convention is to separate each summary with a newline
def three_sentence_summary(text):
    return "\n".join(nltk.tokenize.sent_tokenize(text)[:3])


print(three_sentence_summary(books_dataset["train"][1]["review_body"]))


# Now we implement a function that extracts these summaries from a dataset and
# computes the ROUGE scores for the baseline:
def evaluate_baseline(dataset, metric):
    summaries = [three_sentence_summary(text) for text in dataset["review_body"]]
    return metric.compute(predictions=summaries, references=dataset["review_title"])


# Compute the ROUGE scores over the validation set:
score = evaluate_baseline(books_dataset["validation"], rouge_score)
rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
rouge_dict = dict((rn, round(score[rn] * 100, 2)) for rn in rouge_names)
print(rouge_dict)

# Now we move on to fine-tuning the model with the Trainer API
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
batch_size = 8
# batch_size = 6
# batch_size = 4
num_train_epochs = 8
# Show the training loss with every epoch
logging_steps = len(tokenized_datasets["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    output_dir=f"{model_name}-finetuned-amazon-en-es",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    # Set to indicate we should generate summaries during evaluation so we can compute
    # ROUGE scores for each epoch
    predict_with_generate=True,
    logging_steps=logging_steps,
    # CUDA does not work for this on my machine...
    no_cuda=True,
    push_to_hub=False,
)


# Create a `compute_metrics` function to evaluate our model during training.
# Since we need to `decode` the outputs and labels into text before we compute the
# ROUGE scores this is a little more involved.
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Decode generated summaries into text
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # Decode reference summaries into text
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # ROUGE expects a newline after every sentence
    decoded_preds = [
        "\n".join(nltk.tokenize.sent_tokenize(pred.strip())) for pred in decoded_preds
    ]
    decoded_labels = [
        "\n".join(nltk.tokenize.sent_tokenize(label.strip()))
        for label in decoded_labels
    ]
    # Compute ROUGE scores
    result = rouge_score.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract the scores
    result = {key: round(value * 100, 4) for key, value in result.items()}
    return result


# Now define the data collator for our sequence to sequence task.
# Since mT5 is an encoder-decoder Transformer model, when preparing our batches
# during decoding we need to shift the labels to the right by one.
# This allows the decoder to only see previous ground truth labels and not the
# current or future ones, which would be easy for the model to memorize
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Test out the collator on a small batch of examples
tokenized_datasets = tokenized_datasets.remove_columns(
    books_dataset["train"].column_names
)
# The collator expects a list of dicts where each dict represents a single
# example in the dataset.
features = [tokenized_datasets["train"][i] for i in range(2)]
print(data_collator(features))

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
# This seems like it would take a __cool__ 8+ hours to complete on my machine
# so gonna pass on this...
# trainer.train()
# Now see how the training performed
# print(trainer.evaluate())

# We can also use our accelerated model:
am = AcceleratedMT5(
    datasets=tokenized_datasets,
    data_collator=data_collator,
    model_checkpoint=model_checkpoint,
    tokenizer=tokenizer,
)
# Again this does not work on my GPU...
# am.execute()

# Let's test out how it works:
hub_model_id = "huggingface-course/mt5-small-finetuned-amazon-en-es"
summarizer = pipeline("summarization", model=hub_model_id)


def print_summary(idx):
    review = books_dataset["test"][idx]["review_body"]
    title = books_dataset["test"][idx]["review_title"]
    summary = summarizer(books_dataset["test"][idx]["review_body"])[0]["summary_text"]
    print(f"'>>> Review: {review}'")
    print(f"\n'>>> Title: {title}'")
    print(f"\n'>>> Summary: {summary}'")


print_summary(100)
print_summary(101)
