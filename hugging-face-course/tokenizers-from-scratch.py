from datasets import load_dataset
from tokenizers import (
    Regex,
    Tokenizer,
    decoders,
    models,
    normalizers,
    pre_tokenizers,
    processors,
    trainers,
)
from transformers import PreTrainedTokenizerFast

# train our new tokenizer using a small corpus of text
dataset = load_dataset("wikitext", name="wikitext-2-raw-v1", split="train")


# yield batches of 1000 texts to train the tokenizer
def get_training_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["text"]


# Building a WordPiece tokenizer from scratch
# first instnatiate a tokenizer with a model, set its normalizer, pre_tokenizer
# post_processor, and decoder.
tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# Start with normalization
tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True)

# Here is how you could create the BERT normalizer by hand by composing
# several existing normalizers using a sequence
tokenizer.normalizer = normalizers.Sequence(
    [
        # NFD Unicode normalizer
        normalizers.NFD(),
        normalizers.Lowercase(),
        normalizers.StripAccents(),
    ]
)

# Test out the normalization:
print(tokenizer.normalizer.normalize_str("Héllò hôw are ü?"))

# Next is pre-tokenization
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
# or build it from scratch
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# Or compose multiple
tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
    [pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()]
)

# Next we run inputs through the model. Here we pass through all the
# special tokens we intend to use
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25_000, special_tokens=special_tokens)

# Train our model using the iterator we defined earlier
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Test the tokenizer
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)

# We receive back an encoding, and the last step is to perform post-processing.
# We add tokens to the start and end (or between sentences)
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)

# We use a `TemplateProcessor` to specify how to treat sentences
# The first sentence is represented by $A and the second is $B
# For each of the tokens and sentences we specify the corresponding token type
# ID after a colon
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

# Now our encoding will have the special tokens applied
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)
# And now on a pair of sentences
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences.")
print(encoding.tokens)
print(encoding.type_ids)

# Now finally include a decoder
tokenizer.decoder = decoders.WordPiece(prefix="##")
# And test on our previous encoding
print(tokenizer.decode(encoding.ids))

# Now we can create a fast tokenizer like so:
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    # tokenizer_file="tokenizer.json", # You can load from the tokenizer file, alternatively
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# Now we build a BPE tokenizer from scratch
tokenizer = Tokenizer(models.BPE())

# GPT-2 does not use a normalizer so we go directly to pre-tokenization
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

# For training the only special token for GPT-2 is the end-of-text token
trainer = trainers.BpeTrainer(vocab_size=25_000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Now check the sample text output
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)

# Apply byte-level post-processing for GPT-2 tokenizer
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

sentence = "Let's test this tokenizer."
encoding = tokenizer.encode(sentence)
start, end = encoding.offsets[4]
print(sentence[start:end])

# Add a byte-level decoder
tokenizer.decoder = decoders.ByteLevel()
print(tokenizer.decode(encoding.ids))

# And create the fast tokenizer:
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<|endoftext|>",
    eos_token="<|endoftext|>",
)

# We finish by building a Unigram tokenizer from scratch
tokenizer = Tokenizer(models.Unigram())

# The normalization replacements come from SentencePiece
tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)

tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

# Train the model
special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
trainer = trainers.UnigramTrainer(
    vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>"
)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Look at the sample text
encoding = tokenizer.encode("Let's test this tokenizer.")
print(encoding.tokens)

# Prepare the post-processing
tokenizer.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
)

# Test on a pair of sentences
encoding = tokenizer.encode("Let's test this tokenizer...", "on a pair of sentences!")
print(encoding.tokens)
print(encoding.type_ids)

# Add the decoder
tokenizer.decoder = decoders.Metaspace()
print(tokenizer.decode(encoding.ids))

# And then convert it to a fast tokenizer and may save it for later!
wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
    padding_side="left",
)
