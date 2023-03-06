import collections

import evaluate
import numpy as np
import torch
from accelerated_models import AcceleratedBertSquad
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    pipeline,
)

# We will look into extractive question answering which involves posing questions
# about a document and identifying the answers as spans of text in the document
# itself.
# We fine tune a BERT model on the SQuAD dataset which are questions from
# crowdworkers on a set of Wikipedia articles
raw_dataset = load_dataset("squad")
print(raw_dataset)

print("Context: ", raw_dataset["train"][0]["context"])
print("Question: ", raw_dataset["train"][0]["question"])
print("Answer: ", raw_dataset["train"][0]["answers"])

# There is only one possible answer to every question in our dataset
print(raw_dataset["train"].filter(lambda x: len(x["answers"]["text"]) != 1))
# For evaluation there exists several possible answers for each sample which could
# be the same or be different
print(raw_dataset["validation"][0]["answers"])
print(raw_dataset["validation"][2]["answers"])
# The way our evaluation will work is compare a predicted answer to all the acceptable
# answers and take the best score

# When preprocessing the training data we need to generate labels for the question's
# answer which will be the start and end positions of the tokens corresponding to the
# answer inside the context
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Lets check a sample
context = raw_dataset["train"][0]["context"]
question = raw_dataset["train"][0]["question"]
inputs = tokenizer(question, context)
print(tokenizer.decode(inputs["input_ids"]))

# To deal with long contexts we will limit the length to 100 and use a sliding window
# of 50 tokens
inputs = tokenizer(
    question,
    context,
    max_length=100,
    # truncate the context when the question with its context is too long
    truncation="only_second",
    # number of overlapping tokens between two successive chunks
    stride=50,
    # we want the overflowing tokens
    return_overflowing_tokens=True,
    # we return mappings so we can map to token indices
    return_offsets_mapping=True,
)
for ids in inputs["input_ids"]:
    print(tokenizer.decode(ids))
print(inputs.keys())

# We will only get 0's here since there is only 1 example
print(inputs["overflow_to_sample_mapping"])
inputs = tokenizer(
    raw_dataset["train"][2:6]["question"],
    raw_dataset["train"][2:6]["context"],
    max_length=100,
    # truncate the context when the question with its context is too long
    truncation="only_second",
    # number of overlapping tokens between two successive chunks
    stride=50,
    # we want the overflowing tokens
    return_overflowing_tokens=True,
    # we return mappings so we can map to token indices
    return_offsets_mapping=True,
)
print(f"The 4 examples gave {len(inputs['input_ids'])} features.")
print(f"Here is where each comes from: {inputs['overflow_to_sample_mapping']}.")

# We will map each feature with a corresponding label:
# - (0,0) if the answer is not in the corresponding span of the context
# - (start_pos, end_pos) if the answer is in the corresponding span of the context,
#   with the `start_pos` being the index of the token (in the input IDs) at the start
#   of the answer and the `end_pos` being the index of the token (in th input IDs)
#   where the answer ends

# Find the indices that start and end the context in the input IDs. We use `sequence_ids()`
# method of the `BatchEncoding` returned by the tokenizer in order to compute this
# Once we have the token indices, look at corresponding offsets and detect if the chunk in
# this feature starts after the answer ends or before the answer begins
answers = raw_dataset["train"][2:6]["answers"]
start_pos = []
end_pos = []
for i, offset in enumerate(inputs["offset_mapping"]):
    sample_idx = inputs["overflow_to_sample_mapping"][i]
    answer = answers[sample_idx]
    start_char = answer["answer_start"][0]
    end_char = answer["answer_start"][0] + len(answer["text"][0])
    sequence_ids = inputs.sequence_ids(i)

    # Find the start and end of the context
    idx = 0
    while sequence_ids[idx] != 1:
        idx += 1
    context_start = idx
    while sequence_ids[idx] == 1:
        idx += 1
    context_end = idx - 1

    # If the answer is not fully inside the context the label is (0, 0)
    if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
        start_pos.append(0)
        end_pos.append(0)
    else:
        # Otherwise it's the start and end positions
        idx = context_start
        while idx < context_end and offset[idx][0] <= start_char:
            idx += 1
        start_pos.append(idx - 1)

        idx = context_end
        while idx >= context_start and offset[idx][1] >= end_char:
            idx -= 1
        end_pos.append(idx + 1)

print(start_pos, end_pos)
# Verify the approach is correct:
idx = 0
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

start = start_pos[idx]
end = end_pos[idx]
labeled_answer = tokenizer.decode(inputs["input_ids"][idx][start : end + 1])
print(f"Theoretical answer: {answer}, labels give: {labeled_answer}")

# And check index 4 where we set the labels to (0, 0) where the answer is not
# in the context chunk of the feature
idx = 4
sample_idx = inputs["overflow_to_sample_mapping"][idx]
answer = answers[sample_idx]["text"][0]

decoded_example = tokenizer.decode(inputs["input_ids"][idx])
print(f"Theoretical answer: {answer},\nDecoded example: {decoded_example}")

# Now we preprocess against the whole training dataset. We pad ever feature to the
# maximum length we set, as most contexts will be long, so there is no benefit from
# dynamic padding here
max_length = 384
stride = 128


def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_pos = []
    end_pos = []
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context the label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_pos.append(0)
            end_pos.append(0)
        else:
            # Otherwise it's the start and end positions
            idx = context_start
            while idx < context_end and offset[idx][0] <= start_char:
                idx += 1
            start_pos.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_pos.append(idx + 1)

    inputs["start_positions"] = start_pos
    inputs["end_positions"] = end_pos
    return inputs


# Any apply to the whole dataset:
train_dataset = raw_dataset["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_dataset["train"].column_names,
)
print(len(raw_dataset["train"]), len(train_dataset))


# Preprocessing the validation data is easier since we don't need to generate labels
# so we interpret the predictions of the model into spans of the original context.
# Store both of the offset mappings and match each created feature to the original
# example that it comes from
# We clean up offset mappings related to the question and we set them to `None` since
# we will have no way to know which parts of the input IDs correspond to either the
# question or the context
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []
    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


# Apply the function on the whole validation dataset
validation_dataset = raw_dataset["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_dataset["validation"].column_names,
)
print(len(raw_dataset["validation"]), len(validation_dataset))

# For the trainer we do not use a data collator since we padded all samples to a
# maximum length, and we really just focus on the metric computation
# The steps we will take are:
# 1) Mask start and end logits corresponding to tokens outside of the context
# 2) Convert the start and end logits into probabilities using a softmax
# 3) Attribte a score to each (start_token, end_token) pair by taking the product
#   of the corresponding two probabilities
# 4) Look for the pair with the maximum score that yields a valid answer
# We can skip the softmax step since we don't need to compute scores
# We also don't score all the possible pairs, but just the ones with the highest
# `n_best` logits (with `n_best=20`). Since we skip softmax, those scores will
# be logit scores and will be obtained by taking sum of the start and end logits

# Generate some predictions on a small part of the validation set
small_eval_set = raw_dataset["validation"].select(range(100))
trained_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_dataset["validation"].column_names,
)

# Now that preprocessing is done, change the tokenizer back to the original
# one that we picked.
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Now remove columns from the `eval_set` not expected by the model and build a
# batch with the small validation set and pass it through the model
eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("torch")

device = torch.device("cpu")
batch = {k: eval_set_for_model[k].to(device) for k in eval_set_for_model.column_names}
trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(
    device
)
with torch.no_grad():
    outputs = trained_model(**batch)

# The `Trainer` will give us predictions as NumPy arrays, so we grab start and end
# logits and convert them to that format:
start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()

# Find the predicted answer for each example in our `small_eval_set`
# We first start by mapping each example in `small_eval_set` to the corresponding
# features in the `eval_set`:
example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(eval_set):
    example_to_features[feature["example_id"]].append(idx)

# Loop through all the examples and, for each example, through all the associated features.
# Look at the logit scores for the `n_best` start and end logits, excluding positions
# that give:
# - an answer not inside the context
# - an answer with negative length
# - an answer that is too long
n_best = 20
max_answer_length = 30
predicted_answers = []
for example in small_eval_set:
    example_id = example["id"]
    context = example["context"]
    answers = []

    not_fully_in_context = 0
    bad_length = 0
    for feature_index in example_to_features[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = eval_set["offset_mapping"][feature_index]

        # Returns the indices that would sort the array and then we take the last 20
        start_indexes = np.argsort(start_logit)[-n_best:].tolist()
        end_indexes = np.argsort(end_logit)[-n_best:].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers that are not fully in the context
                if offsets[start_index] is None or offsets[end_index] is None:
                    not_fully_in_context += 1
                    continue
                # Skip answers with a length that is either < 0 or > max_answer_length
                if (
                    end_index < start_index
                    or end_index - start_index + 1 > max_answer_length
                ):
                    bad_length += 1
                    continue
                answers.append(
                    {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )

    if len(answers) > 0:
        best_answer = max(answers, key=lambda x: x["logit_score"])
        predicted_answers.append(
            {"id": example_id, "prediction_text": best_answer["text"]}
        )
    else:
        predicted_answers.append({"id": example_id, "prediction_text": ""})

# The final format of the predicted answers is the one expected by the metric we use:
# A list of dictionaries with one key for the ID of the example and one key for the
# predicated text
metric = evaluate.load("squad")
# The answers will be displayed in a similar format:
theoretical_answers = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
]
print(predicted_answers[0])
print(theoretical_answers[0])
# And here is the score the metric gives us
print(metric.compute(predictions=predicted_answers, references=theoretical_answers))


# Now run this for the scores against the whole dataset
def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    # if offsets[start_index] is None or offsets[end_index] is None:
                    if any(
                        e is None for e in (offsets[start_index], offsets[end_index])
                    ):
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[
                            offsets[start_index][0] : offsets[end_index][1]
                        ],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in examples
    ]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)


# And check that it works on our predictions:
print(compute_metrics(start_logits, end_logits, eval_set, small_eval_set))

# NOW WE FINE TUNE THE MODEL:
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
)
# Training can take up to an hour on a good GFX card so we omit this here...
"""
trainer.train()

# Evaluate the model
predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions
print(
    compute_metrics(
        start_logits, end_logits, validation_dataset, raw_dataset["validation"]
    )
)
"""

# And we can create a custom accelerated model / training loop
# Again this seems like it could take 1+ hours...
"""
am = AcceleratedBertSquad(
    dataset=raw_dataset,
    train_dataset=train_dataset,
    raw_validation_dataset=raw_dataset["validation"],
    tokenizer=tokenizer,
    model_checkpoint=model_checkpoint,
)
am.execute()
"""

# Use the fine tuned model from the course and check it out!
model_checkpoint = "huggingface-course/bert-finetuned-squad"
question_answerer = pipeline("question-answering", model=model_checkpoint)

context = """
🤗 Transformers is backed by the three most popular deep learning libraries — Jax, PyTorch and TensorFlow — with a seamless integration
between them. It's straightforward to train your models with one before loading them for inference with the other.
"""
question = "Which deep learning libraries back 🤗 Transformers?"
print(question_answerer(question=question, context=context))
