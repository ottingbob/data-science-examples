# Might need to run:
# python -m textblob.download_corpora
# poetry run textblob.download_corpora

import spacy
import torch
from sklearn import svm
from textblob import TextBlob

# Sentiment
phrase = "thhe boook was horraible"

tb_phrase = TextBlob(phrase)
tb_phrase = tb_phrase.correct()

print(tb_phrase)
print(tb_phrase.tags)
print(tb_phrase.sentiment)

# Transformer Architecture
# Recurrent Neural Networks. These allow for context dependent
# neural network where it can learn over time
nlp = spacy.load("en_core_web_md")
doc = nlp("Here is some text to encode.")


class Category:
    BOOKS = "BOOKS"
    BANK = "BANK"


train_x = [
    "good characters and plot progression",
    "check out the book",
    "good story. would recommend",
    "novel recommendation",
    "need to make a deposit to the bank",
    "balance inquiry savings",
    "save money",
]
train_y = [
    Category.BOOKS,
    Category.BOOKS,
    Category.BOOKS,
    Category.BOOKS,
    Category.BANK,
    Category.BANK,
    Category.BANK,
]

docs = [nlp(text) for text in train_x]
train_x_vectors = [doc.vector for doc in docs]
clf_svm = svm.SVC(kernel="linear")

clf_svm.fit(train_x_vectors, train_y)

# The bert model is able to differentiate that we are using `check` in a banking
# context even though our training data only provides to the word in the context
# of a book category.
test_x = ["check this story out", "make big money", "i need to write a check"]
docs = [nlp(text) for text in test_x]
test_x_vectors = [doc.vector for doc in docs]

print(clf_svm.predict(test_x_vectors))
