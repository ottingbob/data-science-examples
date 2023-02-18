from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer


# Define some training utterances
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

# Fit vectorizer to transform text to bag-of-words vectors
vectorizer = CountVectorizer(binary=True)
train_x_vectors = vectorizer.fit_transform(train_x)

print(vectorizer.get_feature_names_out())
print(train_x_vectors.toarray())

# Train SVM Model
clf_svm = svm.SVC(kernel="linear")
clf_svm.fit(train_x_vectors, train_y)

# Test new utterances on trained model
test_x = vectorizer.transform(["i love the books"])
# The model doesn't know the word `books` =(
print(clf_svm.predict(test_x))
