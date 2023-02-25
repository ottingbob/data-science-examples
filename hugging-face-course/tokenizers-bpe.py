from collections import defaultdict
from functools import wraps
from time import time

from transformers import AutoTokenizer


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__} args:[{args}, {kw}] took: {te-ts:2.6f} sec")
        return result

    return wrap


# Implementing BytePair Encoding algorithm
corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

tokenizer = AutoTokenizer.from_pretrained("gpt2")

word_freqs = defaultdict(int)

# Get frequencies of each word in the corpus (pre-tokenization)
for text in corpus:
    words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
        text
    )
    new_words = [word for word, offset in words_with_offsets]
    for word in new_words:
        word_freqs[word] += 1

print(word_freqs)


# Compute the base vocabulary, formed by all the characters used in the corpus:
@timing
def compute_base_vocab():
    alphabet = []
    for word in word_freqs.keys():
        for letter in word:
            if letter not in alphabet:
                alphabet.append(letter)
    return alphabet


@timing
def compute_base_vocab_lc():
    return list(set([letter for word in word_freqs.keys() for letter in word]))


compute_base_vocab()
alphabet = compute_base_vocab_lc()
alphabet.sort()
print(alphabet)

# Add special tokens used by the model at the beginning of the vocabulary.
# For GPT-2, the only special token is "<|endoftext|>"
vocab = ["<|endoftext|>"] + alphabet.copy()

# Split each word into individual characters to start training
splits = {word: [c for c in word] for word in word_freqs.keys()}


# Compute the frequency of each pair for each step of training
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1:
            continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


pair_freqs = compute_pair_freqs(splits)
for i, key in enumerate(pair_freqs.keys()):
    print(f"{key}: {pair_freqs[key]}")
    if i >= 5:
        break

# Find the most frequent pair
best_pair = ""
max_freq = None
for pair, freq in pair_freqs.items():
    if max_freq is None or max_freq < freq:
        best_pair = pair
        max_freq = freq
print(f"Best pair: {best_pair} Max Freq: {max_freq}")
# Best pair: ('Ġ', 't') Max Freq: 7

# So the first merge to learn is ('Ġ', 't') -> 'Ġt',
# and we add 'Ġt' to the vocabulary:
merges = {("Ġ", "t"): "Ġt"}
vocab.append("Ġt")


# We need to apply that merge in our splits dictionary
def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        if len(split) == 1:
            continue

        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2 :]
            else:
                i += 1
        splits[word] = split
    return splits


splits = merge_pair("Ġ", "t", splits)
# Here are the results of the first merge:
print(splits["Ġtrained"])
# ['Ġt', 'r', 'a', 'i', 'n', 'e', 'd']

# Loop until we have learned all the merges we want, here we go for
# a vocab size of 50:
vocab_size = 50
while len(vocab) < vocab_size:
    pair_freqs = compute_pair_freqs(splits)
    best_pair = ""
    max_freq = None
    for pair, freq in pair_freqs.items():
        if max_freq is None or max_freq < freq:
            best_pair = pair
            max_freq = freq
    splits = merge_pair(*best_pair, splits)
    merges[best_pair] = best_pair[0] + best_pair[1]
    vocab.append(best_pair[0] + best_pair[1])
print(merges)
print(vocab)


# Tokenize new text, pre-tokenize it, split it, apply all merge rules:
def tokenize(text):
    pre_tokenize_result = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, offset in pre_tokenize_result]
    splits = [[l for l in word] for word in pre_tokenized_text]
    for pair, merge in merges.items():
        for idx, split in enumerate(splits):
            i = 0
            while i < len(split) - 1:
                if split[i] == pair[0] and split[i + 1] == pair[1]:
                    split = split[:i] + [merge] + split[i + 2 :]
                else:
                    i += 1
            splits[idx] = split
    return sum(splits, [])


# Try it out on some text:
print(tokenize("This is not a token."))
