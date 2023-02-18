import json
import os
from dataclasses import dataclass
from pprint import pprint
from typing import List

import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from textblob import TextBlob


# Load in the data
@dataclass
class Review:
    category: str
    text: str


class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
        self.stop_words = stopwords.words("english")
        self.lemmatizer = WordNetLemmatizer()

    @property
    def text(self) -> List[str]:
        # Modify the text here in order to remove stopwords
        text_list = []
        for r in self.reviews:
            # TextBlob really tanks the performance here...
            # r_text = TextBlob(r.text).correct()
            phrase = [
                self.lemmatizer.lemmatize(word, pos="v")
                # for word in word_tokenize(r_text.string)
                for word in word_tokenize(r.text)
                if word not in self.stop_words
            ]
            text_list.append(" ".join(phrase))
        # return [x.text for x in self.reviews]
        return text_list

    @property
    def y(self) -> List[str]:
        return [x.category for x in self.reviews]


train_reviews = []
all_categories = []


def get_file_data(
    directory: str,
    file_prefix: str,
) -> List[Review]:
    reviews = []
    for file in os.listdir(directory):
        category = file.strip(file_prefix).split(".")[0]
        all_categories.append(category)
        with open(f"{directory}/{file}") as f:
            for line in f:
                review_json = json.loads(line)
                review = Review(category, review_json["reviewText"])
                reviews.append(review)
    return reviews


print("Collecting training data...")
train_reviews = get_file_data("./data/training", "train_")
train_container = ReviewContainer(train_reviews)

# all_categories will get duplicated so we just reinstantiate it here
all_categories = []

print("Collecting testing data...")
test_reviews = get_file_data("./data/test", "test_")
test_container = ReviewContainer(test_reviews)


def bag_of_words_model():
    # Train model with Bag of words
    print("Training bag of words model...")
    corpus = train_container.text
    vectorizer = CountVectorizer(binary=True)

    # training text converted into vector
    train_x = vectorizer.fit_transform(corpus)

    clf = svm.SVC(kernel="linear")
    clf.fit(train_x, train_container.y)

    # Evaluate performance
    # make sure to convert test text to vector form
    print("Testing data against model...")
    test_corpus = test_container.text
    test_x = vectorizer.transform(test_corpus)

    # Overall Accuracy 0.6542222222222223
    print("Overall Accuracy", clf.score(test_x, test_container.y))

    y_pred = clf.predict(test_x)

    print("f1 scores by category")
    pprint(
        list(
            zip(
                all_categories,
                f1_score(test_container.y, y_pred, average=None, labels=all_categories),
            )
        )
    )


# FIXME: This takes like 5 mins to load lol...
def word_vector_model():
    nlp = spacy.load("en_core_web_md")

    # Word vectorizer model
    print("Training word vectorizer model...")
    corpus = [nlp(text) for text in train_container.text]
    train_x_word_vectors = [x.vector for x in corpus]

    clf_wv = svm.SVC(kernel="linear")
    clf_wv.fit(train_x_word_vectors, train_container.y)

    # Evaluate performance
    # make sure to convert test text to vector form
    print("Testing data against model...")
    test_corpus = [nlp(text) for text in test_container.text]
    test_x_word_vectors = [x.vector for x in test_corpus]

    # Overall Accuracy 0.6746666666666666
    print("Overall Accuracy", clf_wv.score(test_x_word_vectors, test_container.y))


bag_of_words_model()
# TODO: Run against a fine tuned BERT model
# https://explosion.ai/blog/spacy-transformers
# TODO: Part of speech tagging
