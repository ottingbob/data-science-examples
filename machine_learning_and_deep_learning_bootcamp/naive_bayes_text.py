from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]

training_data = fetch_20newsgroups(
    subset="train",
    categories=categories,
    shuffle=True,
    # random_state=None,
)

print("\n".join(training_data.data[0].split("\n")[:10]))
print("Target is:", training_data.target_names[training_data.target[1]])

# TfidfVectorizer = CountVectorizer + TfidfTransformer

# Count the word occurrences
count_vector = CountVectorizer()
x_train_counts = count_vector.fit_transform(training_data.data)

# We transform the word occurrences into tf-idf
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)

# Now we use the multi-nomial gaussian naive bayes classifier to make predictions
model = MultinomialNB().fit(x_train_tfidf, training_data.target)

# Test sentences which do not fit into the categories that we have chosen above
new = [
    "My favorite topic has something to do with quantum physics and quantum mechanics",
    "This has nothing to do with church or religion",
    "Software engineering is getting hotter and hotter nowadays",
]
# Convert the sentences into numerical values to make predictions
x_new_counts = count_vector.transform(new)
x_new_tfidf = tfidf_transformer.transform(x_new_counts)

predicted = model.predict(x_new_tfidf)

print("\nPredicted output:\n" + "=" * 20)
for doc, category in zip(new, predicted):
    print(f"{doc} ---> {training_data.target_names[category]}")
