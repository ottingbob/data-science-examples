import collections
from typing import Dict, List

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Stop words are common words which do not add any meaning to a sentence
# (from an LLM point of view)
# Words such as: ["I", "am", "he", "she", "is", "on", "in"]
nltk.download("stopwords")


def tokenizer(text):
    # Transform the text into an array of words
    tokens = word_tokenize(text)
    # Yields the stem or base of a given word: (fishing-fish, fisher-fish)
    stemmer = PorterStemmer()
    # Filter out the stopwords
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words("english")]
    return tokens


def cluster_sentences(sentences: List[str], n_clusters: int = 2) -> Dict:
    # Create TF-IDF again: stopwords -> we filter out common words
    vectorizer = TfidfVectorizer(
        tokenizer=tokenizer,
        stop_words=stopwords.words("english"),
        # Transform everything to lowercase
        lowercase=True,
    )
    # Build a TF-IDF matrix for the sentences
    matrix = vectorizer.fit_transform(sentences)
    # Fit the K-Means clustering model
    model = KMeans(n_clusters=n_clusters)
    model.fit(matrix)
    topics = collections.defaultdict(list)

    for index, label in enumerate(model.labels_):
        topics[label].append(index)

    return dict(topics)


if __name__ == "__main__":
    # We don't have training labels so we are using unsupervised learning with
    # an unsupervised algorithm: KMeans
    sentences = [
        "Quantum physics is quite important in science nowadays.",
        "Software engineering is hotter and hotter topic in the silicon valley",
        "Investing in stocks and trading with them are not that easy",
        "FOREX is the stock market for trading currencies",
        "Warren Buffet is famous for making good investments. He knows stock markets",
    ]

    n_clusters = 2
    clusters = cluster_sentences(sentences, n_clusters)

    for cluster in range(n_clusters):
        print("CLUSTER ", cluster + 1, ":")
        for i, sentence in enumerate(clusters[cluster]):
            print("\tSENTENCE ", i + 1, ": ", sentences[sentence])
