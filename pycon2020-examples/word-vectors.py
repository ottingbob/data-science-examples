from pprint import pprint

import spacy
from sklearn import svm


class Category:
    BOOKS = "BOOKS"
    CLOTHING = "CLOTHING"


train_x = [
    "i love the book",
    "this is a great book",
    "the fit is great",
    "i love the shoes",
]
train_y = [Category.BOOKS, Category.BOOKS, Category.CLOTHING, Category.CLOTHING]

nlp = spacy.load("en_core_web_md")

print(train_x)

# word vector representation of our train sentences above
docs = [nlp(text) for text in train_x]
train_x_word_vectors = [x.vector for x in docs]

clf_svm_wv = svm.SVC(kernel="linear")
clf_svm_wv.fit(train_x_word_vectors, train_y)

test_x = [
    "I love the story",
    "I love the hat",
    "these earrings hurt",
    "I went to the bank and wrote a check",
    "let me check that out",
]
test_docs = [nlp(text) for text in test_x]
test_x_word_vectors = [x.vector for x in test_docs]

pprint(list(zip(test_x, clf_svm_wv.predict(test_x_word_vectors))))
