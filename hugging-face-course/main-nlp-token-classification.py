import evaluate
import numpy as np
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    get_scheduler,
    pipeline,
)

# This is a dataset that contains news stories from Reuters which contains
# labels for `NER`, `POS`, and chunking
raw_datasets = load_dataset("conll2003")

print(raw_datasets["train"][0]["tokens"])
# To perform NER look at the related tags
print(raw_datasets["train"][0]["ner_tags"])
# And access the correspondence between the tagged integers and the label
# names by looking at the features of the dataset
ner_feature = raw_datasets["train"].features["ner_tags"]
print(ner_feature)
label_names = ner_feature.feature.names
print(label_names)

# Decode the labels
words = raw_datasets["train"][0]["tokens"]
labels = raw_datasets["train"][0]["ner_tags"]
line1 = ""
line2 = ""
for word, label in zip(words, labels):
    full_label = label_names[label]
    max_length = max(len(word), len(full_label))
    line1 += word + " " * (max_length - len(word) + 1)
    line2 += full_label + " " * (max_length - len(full_label) + 1)

print(line1)
print(line2)

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
inputs = tokenizer(raw_datasets["train"][0]["tokens"], is_split_into_words=True)
print(inputs.tokens())
# Map each token to a corresponding word
print(inputs.word_ids())


# Expand our label list to match the tokens. We apply special tokens with a label
# of -100 because by default -100 is an index that is ignored in the loss function
# we will use (cross entropy).
# Next each token gets the same label as the token that started the word it's inside
# since they are a part of the same entity. For tokens inside a word, but not at the
# start we replace `B-` with `I-`
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)
    return new_labels


# Try it out on the first sentence
labels = raw_datasets["train"][0]["ner_tags"]
word_ids = inputs.word_ids()
print(labels)
print(align_labels_with_tokens(labels, word_ids))


# Apply to the whole dataset by tokenizing all inputs and applying on all
# the labels:
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# Apply the preprocessing in one go on the other splits of our dataset
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

# Next we fine tune the Trainer with specified batches and a metric
# computation function
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
# Test on a few samples
batch = data_collator([tokenized_datasets["train"][i] for i in range(2)])
print(batch["labels"])
# Compare this to the labels for the first and second elements in our dataset
for i in range(2):
    print(tokenized_datasets["train"][i]["labels"])

# Now we word on the Trainer to compute a metric for every epoch by defining
# a compute_metrics function that takes an array of predictions and labels
# and returns a dictionary with the metric names and values
# A framework used to evaluate token classification prediction is called
# `seqeval` and we load it like so:
metric = evaluate.load("seqeval")

# The metric is backwards and does not work like standard accuracy. Instead
# it takes the list of labels as strings (NOT INTEGERS) so we need to decode
# the predications and labels before passing them to the metric
# Here is an example
labels = raw_datasets["train"][0]["ner_tags"]
labels = [label_names[i] for i in labels]
print(labels)

# Simulate fake predictions by changing the label at index 2
predictions = labels.copy()
predictions[2] = "0"
computed = metric.compute(predictions=[predictions], references=[labels])
print(computed)


# Our compute_metrics function takes the argmax of the logits to convert
# them to predictions, then we convert both labels and predictions from
# integers to strings. We remove all the values where the label is `-100`
# then pass the results to the `metric.compute()` method
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    # Remove ignored index (special tokens) and covert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


# Now define the model to fine-tune
# We need to be able to pass some information on the number of labels we have
# We pass that mapping with the `id2label` and `label2id` mappings below
id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
# Make sure we have the right number of labels on the model
assert model.config.num_labels == 9

# NOW WE TRAIN THE MODEL
args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

# trainer.train()
# It took about 26 minutes to train for me:
# {'train_runtime': 1549.8165, 'train_samples_per_second': 27.179, 'train_steps_per_second': 3.399, 'train_loss': 0.06669504794553092, 'epoch': 3.0}
# 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5268/5268 [25:49<00:00,  3.40it/s]
# Saving model checkpoint to bert-finetuned-ner/checkpoint-5268

# Now test it out
model_checkpoint = "bert-finetuned-ner/checkpoint-5268"
token_classifier = pipeline(
    "token-classification", model=model_checkpoint, aggregation_strategy="simple"
)
result = token_classifier("My name is Sylvain and I word at Hugging Face in Brooklyn.")
print(result)
# INTERESTING: In our model it classifies `Hugging Face` as a location
# [{'entity_group': 'PER', 'score': 0.9984589, 'word': 'Sylvain', 'start': 11, 'end': 18}, {'entity_group': 'LOC', 'score': 0.8257878, 'word': 'Hugging Face', 'start': 33, 'end': 45}, {'entity_group': 'LOC', 'score': 0.9973863, 'word': 'Brooklyn', 'start': 49, 'end': 57}]

# After getting the custom training loop to not have memory issues it seems like the
# integrated GPU on my machine is going to end up taking a little longer than using the
# method shown above:
#
# You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
# 18%|█████████████████████▎| 1936/10533 [06:05<27:56,  5.13it/s]
#
# Here are the results at the first epoch. I stopped after this:
# ...

# Additionally we can test out a custom training loop
# Make the batch size smaller to avoid:
# torch.cuda.OutOfMemoryError: CUDA out of memory.
batch_size = 4
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=batch_size,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    batch_size=batch_size,
)
# Start with a clean model to not pick up any training that may have happened previously
model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    id2label=id2label,
    label2id=label2id,
)
optimizer = AdamW(model.parameters(), lr=2e-5)
accelerater = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerater.prepare(
    model,
    optimizer,
    train_dataloader,
    eval_dataloader,
)
# After preparing our dataloader we need to compute the training steps using
# its new length. The preparation will change the length so we make sure to account
# for this.
num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# Now write the full training loop.
# we simplify the evaluation by defining `postprocess` here which takes predictions
# and labels and coverts them to lists of strings, like our `metric` object expects
def postprocess(predictions, labels):
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens) and convert to labels
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


# The training loop has 3 parts:
# 1) Training which is classic iteration over the train_dataloader, forward pass
# through the model, then backward pass and optimizer step
# 2) Evaluation where we pad predictions and labels to the same shape and send
# results to metric computation
# 3) Saving and uploading -- WE DONT DO THIS GOOFY STEP!!!
progress_bar = tqdm(range(num_training_steps))
for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerater.backward(loss)

        optimizer.step()
        # LOL this did not work either...
        # Try to delete intermediate vars after optimizer step
        # del outputs, loss
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        # Set in an attempt to fix the error:
        # torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 3.81 GiB total capacity; 2.86 GiB already allocated; 448.00 KiB free; 3.09 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.  See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
        #
        # Emptying the cache below simply did not fix it so I tried using the below:
        # export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
        #
        # Tried running the program like so:
        # PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 python hugging-face-course/main-nlp-token-classification.py
        torch.cuda.empty_cache()

    # Evaluation
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]

        # Pad predictions and labels for being gathered
        # Got REKT here:
        """
        File "/home/yobawb/dev/python/data-science-examples/hugging-face-course/main-nlp-token-classification.py", line 330, in <module>
            predictions, dim=1, pad_index=-100
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
          File "/home/yobawb/.pyenv/versions/data-science-examples/lib/python3.11/site-packages/evaluate/module.py", line 474, in add_batch
            raise ValueError(
            ValueError: Bad inputs for evaluation module: ['labels']. All required inputs are ['predictions', 'references']
             33%|██████████████████████████████████████▋                                                                             | 3511/10533 [11:06<22:13,  5.27it/s]
        )
        """
        predictions = accelerater.pad_across_processes(
            predictions, dim=1, pad_index=-100
        )
        labels = accelerater.pad_across_processes(labels, dim=1, pad_index=-100)

        predictions_gathered = accelerater.gather(predictions)
        labels_gathered = accelerater.gather(labels)

        true_predictions, true_labels = postprocess(
            predictions_gathered, labels_gathered
        )
        metric.add_batch(predictions=true_predictions, labels=true_labels)

    results = metric.comput()
    print(
        f"epoch: {epoch}",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

    # Optionally save at each epoch
    accelerater.wait_for_everyone()
    """
    unwrapped_model = accelerater.unwrap_model(model)
    unwrapped_model.save_pretrained(OUTPUT_DIR, save_function=accelerater.save)
    if accelerater.is_main_process:
        tokenizer.save_pretrained(OUTPUT_DIR)
        # PUSH TO HUB IF YOU SO CHOOSE
    """

# Again -- test the accelerated model out
unwrapped_model = accelerater.unwrap_model(model)
token_classifier = pipeline(
    "token-classification", model=unwrapped_model, aggregation_strategy="simple"
)
result = token_classifier("My name is Sylvain and I word at Hugging Face in Brooklyn.")
print(result)
