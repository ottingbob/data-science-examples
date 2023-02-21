import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    get_scheduler,
)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = [
    "I've been waiting for a HuggingFace course my whole life.",
    "This course is amazing!",
]
batch = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

batch["labels"] = torch.tensor([1, 1])

optimizer = AdamW(model.parameters())
loss = model(**batch).loss
loss.backward()
optimizer.step()

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

raw_train_dataset = raw_datasets["train"]
print(raw_train_dataset[0])
print(raw_train_dataset.features)

tokenized_sentences_1 = tokenizer(raw_train_dataset["sentence1"])
tokenized_sentences_2 = tokenizer(raw_train_dataset["sentence2"])

# The `token_type_ids` tell the model which part of the input is the first
# sentence and which is the second sentence
inputs = tokenizer("This is the first sentence.", "This is the second one.")
print(inputs)
print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))

# This will store the entire dataset in memory and process it rather slowly:
"""
tokenized_dataset = tokenizer(
    raw_train_dataset["sentence1"],
    raw_train_dataset["sentence2"],
    padding=True,
    truncation=True,
)
"""


# To keep the data as a dataset, we can use `Dataset.map()` by applying a function
# on each element of the dataset. This one will tokenize our inputs
def tokenize_func(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_func, batched=True)
print(tokenized_datasets)

# Last thing we do is pad all the examples to the length of the longest element
# when we batch elements together -- called dynamic padding
# To do this in practice we define a collate function that will apply the correct
# amount of padding to the items of the dataset we want to batch together.
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

samples = tokenized_datasets["train"][:8]
# extract the keys `label`, `input_ids`, `token_type_ids`, `attention_mask`
samples = {
    k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]
}
print([len(x) for x in samples["input_ids"]])

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})


# Compare the predictions to the labels using metrics from the `evaluate` library:
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train_with_trainer():
    # Next we fine-tune the model using the `Trainer` class
    training_args = TrainingArguments(
        "test-trainer", eval_steps=10, evaluation_strategy="steps"
    )
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        # the `data_collator` arg defaults to a DataCollatorWithPadding so this is redundant
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # We can evaluate the model by building a `compute_metrics` function and use it the
    # next time we train
    predictions = trainer.predict(tokenized_datasets["validation"])
    print(predictions.predictions.shape, predictions.label_ids.shape)

    preds = np.argmax(predictions.predictions, axis=-1)
    print(preds)


# Train using the trainer class provided by transformers:
# train_with_trainer()


def train_with_pytorch():
    tokenized_datasets = raw_datasets.map(tokenize_func, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(
        ["sentence1", "sentence2", "idx"]
    )
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    # Check that the result only has columns that our model will accept
    print(tokenized_datasets["train"].column_names)

    accelerator = Accelerator()

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=8,
        collate_fn=data_collator,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    # Make sure there are no mistakes in data processing by inspecting a batch
    for batch in train_dataloader:
        break
    batch_shapes = {k: v.shape for k, v in batch.items()}
    print(batch_shapes)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape)

    # Now we need a model and a learning rate scheduler
    # AdamW is the same as Adam but adds in weight decay regularization
    # The learning rate scheduler is a linear decay from the maximum value
    # (5e-5) to 0
    # To define it properly we need to know the number of training steps we will take
    # which is the number of epochs multiplied by the number of training batches
    optimizer = AdamW(model.parameters(), lr=5e-5)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    print(num_training_steps)

    # Specify whether we are using CPU or GPU for training computation
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # model.to(device)
    # print(device)

    # Run the training
    progress_bar = tqdm(range(num_training_steps))
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            # loss.backward()
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    # Evaluate the training
    metric = evaluate.load("glue", "mrpc")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    print(metric.compute())


train_with_pytorch()
