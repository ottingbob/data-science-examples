from sklearn.feature_extraction.text import TfidfVectorizer

vec = TfidfVectorizer()

# Coverts a collection of raw documents to a matrix of TF-IDF features
# TF: Text Frequency
#   How often does a word appear in a document -- positive impact
# IDF: Inverse Document Frequency
#   How often does a word appear across documents -- negative impact
tfidf = vec.fit_transform(
    [
        "I like machine learning and cluster algorithms",
        "Apples, oranges and any kind of fruits are healthy",
        "Is it feasible with machine learning algorithms?",
        "My family is happy because of the healthy fruits",
    ]
)

ans = tfidf * tfidf.T
print(ans.A)
