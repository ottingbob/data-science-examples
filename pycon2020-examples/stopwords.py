from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Stopwords are common words in english such as:
# This, that, he, it, the, these, them, etc.

stop_words = stopwords.words("english")

phrase = "Here is an example sentence demonstrating the removal of stopwords"

words = word_tokenize(phrase)

stripped_phrase = []
for word in words:
    if word not in stop_words:
        stripped_phrase.append(word)

print(" ".join(stripped_phrase))
# Here example sentence demonstrating removal stopwords
