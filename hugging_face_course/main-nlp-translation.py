import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# For this example we will fine-tune a Marian model pretrained to translate
# from English to French on the KDE4 dataset which is a dataset of localized
# files for the KDE apps.
# The model we use has been pretrained on a large corpus of French and English
# texts taken from the Opus dataset (which contains the KDE4 dataset)
raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
print(raw_datasets)

# Create a validation dataset
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
# rename the "test" key to "validation"
split_datasets["validation"] = split_datasets.pop("test")
print(split_datasets)
print(split_datasets["train"][1]["translation"])

model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

# Process one sample of each language in the training set:
en_sentence = split_datasets["train"][1]["translation"]["en"]
fr_sentence = split_datasets["train"][1]["translation"]["fr"]
inputs = tokenizer(en_sentence, text_target=fr_sentence)
print(inputs)

# Define the preprocessing function to apply on datasets
max_length = 128


def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=max_length,
        truncation=True,
    )
    return model_inputs


tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# We need a data collator to deal with the padding for dynamic batching
# and should be padded to the max length found in the labels
# Padding value to pad labels should be `-100` and not the padding of the
# tokenizer to make sure padded values are ignored in the loss computation
#
# It takes the model because this data collator will also be responsible
# for preparing the decoder input IDs, which are shifted versions of the
# labels with a special token at the beginning. Since this shift is done
# slightly differently for different architectures, the DataCollatorForSeq2Seq
# needs to know the model object
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Test on a few samples
batch = data_collator([tokenized_datasets["train"][i] for i in range(1, 3)])
print(batch.keys())

# We use the BLEU score as a metric. It evaluates how close translations are
# to their labels, but does not measure the intelligibility or grammatical
# correctness of the models generated outputs, but uses statistical rules
# to ensure that all the words in the generated outputs also appear in
# the targets. In addition there are rules that penalize repetitions of the
# same words if they are not also repeated in the targets
# We use the `sacrebleu` library which allows us to use a BLEU metric when
# the text is not already tokenized so we can compare scores between models
# of different tokenizers
metric = evaluate.load("sacrebleu")

# Here is an example
predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
print(metric.compute(predictions=predictions, references=references))
# And an example of a BAD prediction
predictions = ["This This This This"]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
print(metric.compute(predictions=predictions, references=references))


# To get model outputs to texts the metric can use, we use the
# `tokenizer.batch_decode()` method. We just have to clean up the `-100s`
# in the labels
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # In case the model returns more than the predication logits
    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100s in the labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Simple post-processing
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


args = Seq2SeqTrainingArguments(
    "marian-finetuned-kde4-en-to-fr",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    # per_device_train_batch_size=32,
    # per_device_eval_batch_size=64,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
)
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Before training look at the score the model gets
# This takes up to 40 minutes...
# print(trainer.evaluate(max_length=max_length))
# trainer.train()

# TODO: Implement a custom training loop...
