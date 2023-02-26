import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
    BertModel,
    BertTokenizer,
    pipeline,
)


def full_example():
    # This pipeline groups together 3 steps:
    # 1) Preprocessing
    # 2) Passing inputs through the model
    # 3) Postprocessing
    classifier = pipeline("sentiment-analysis")
    raw_inputs = [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
    res = classifier(raw_inputs)
    print(res)

    # The tokenizer is responsible for:
    # - splitting the input into words, subwords, or symbols called tokens
    # - mapping each token to an integer
    # - adding additional inputs that may be useful to the model

    # Use the default checkpoint of the `sentiment-analysis` pipeline
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    print(inputs)

    # A vector output by the transformer module is usually large and has 3 dimensions:
    # 1) Batch size: Number of sequences processed at a time (2 in this example)
    # 2) Sequence length: Length of the numerical representation of the sequence (16 in this example)
    # 3) Hidden size: Vector dimensions of each model input
    model = AutoModel.from_pretrained(checkpoint)

    # This is a high-dimensional vector because of the hidden size. This can be very large
    # ranging from 768 for smaller models up to 3072 or more for larger ones.
    outputs = model(**inputs)
    print(outputs.last_hidden_state.shape)

    # Now we need a model with a sequence classification head to classify sentences as
    # positive or negative
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    outputs = model(**inputs)
    # logits are the raw unnormalized scores outputted by the last layer of the model
    print(outputs.logits.shape)
    print(outputs.logits)

    # To be converted to probabilities they need to go through a SoftMax layer
    # All transformers models output logits, as the loss function for training will generally
    # fuse the last activation function, such as SoftMax, with the actual loss function, such
    # as cross entropy.
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    print(predictions)

    # To get the labels corresponding to each position, we can inspect the `id2label`
    # attribute of the model config
    print(model.config.id2label)


# Run the full example:
# full_example()


def manual_model_and_token():
    # Working with a BERT model directly
    config = BertConfig()
    # This shows the different configuration variables that are available
    print(config)

    # Initialize the model with all the weights of the checkpoint
    checkpoint = "bert-base-cased"
    model = BertModel.from_pretrained(checkpoint)

    # Saving a model is simple:
    # model.save_pretrained("directory_location")
    # And will end up saving 2 files to disk:
    # config.json & pytorch_model.bin
    sequences = ["Hello!", "Cool.", "Nice!"]
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoded_sequences = tokenizer(sequences, padding=True, truncation=True).input_ids
    print(encoded_sequences)

    model_inputs = torch.tensor(encoded_sequences)
    print(model_inputs)

    output = model(model_inputs)
    print(output)
    # encoded_sequences = model(**sequences)

    # Here we can start to use the bert tokenizer as well
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    sequence = "Using a Transformer network is simple"
    res = tokenizer(sequence)
    print(res)

    tokens = tokenizer.tokenize(sequence)
    print(tokens)

    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)

    # Decoding is the other way around, from vocabulary indices to strings:
    decoded_string = tokenizer.decode(ids)
    print(decoded_string)


# Run the modeler and tokenizer manually:
# manual_model_and_token()


def handle_multiple_sequences():
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence = "I've been waiting for a HuggingFace course my whole life."

    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    # Add an extra dimension for the model to work properly. I.E. We wrap the
    # `ids` input in an array:
    input_ids = torch.tensor([ids])
    print("Input IDs:", input_ids)
    output = model(input_ids)
    print("Logits:", output.logits)

    # Padding tokens make sure that when you batch sequences together into a tensor
    # they are able to have the same length. Based on the tokenizer there are different
    # padding token values configured.
    sequence1_ids = [[200, 200, 200]]
    sequence2_ids = [[200, 200, 200]]
    batched_ids = [
        [200, 200, 200],
        [200, 200, tokenizer.pad_token_id],
    ]
    print(model(torch.tensor(sequence1_ids)).logits)
    print(model(torch.tensor(sequence2_ids)).logits)
    print(model(torch.tensor(batched_ids)).logits)

    # Attention masks are tensors with the exact same shape as the input IDs tensor,
    # filled with 0s and 1s: 1s indicate the corresponding tokens should be attended
    # to and 0s indicate the tokens should be ignored by the attention layers of
    # the model.
    attention_mask = [
        [1, 1, 1],
        [1, 1, 0],
    ]
    outputs = model(
        torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask)
    )
    print(outputs.logits)

    # Models usually handle sequences of up to 512 or 1024 tokens and crash if larger
    # ones are provided. As a result it is recommended to truncate sequences by specifying
    # a `max_sequence_length` parameter: sequence[:max_sequence_length]


# Run multiple sequences through a model manually:
# handle_multiple_sequences()

# Run a tokenizer that can handle multiple sequences (padding), long sequences (truncation),
# and multiple types of tensors:
checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
print(output)
