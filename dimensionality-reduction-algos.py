# Examples taken from:
# https://machinelearningmastery.com/dimensionality-reduction-algorithms-with-python/
from functools import partial

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def evaluate_model(
    model_name: str,
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
):
    # evaluate model
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    n_scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)
    # report performance
    print(
        f"{model_name}: Accuracy Mean: {np.mean(n_scores):.3f}\tAccuracy STD: ({np.std(n_scores):.3f})"
    )


# Create a classification dataset
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_redundant=10, random_state=7
)
# summarize the dataset
print(X.shape, y.shape)
evaluate = partial(evaluate_model, X=X, y=y)

# Evaluate logistic regression model on raw data
# define model
model = LogisticRegression()
evaluate("Logistic Regression", model)
# Accuracy: 0.824 (0.034)

# A successful dimensionality reduction transform on this data should result in a
# model that has better accuracy than this baseline

# Here are examples of reducing the 20 input columns to 10 where possible

# define pipeline for `Principal Component Analysis`
# this is good for dense data with few zero values
model = Pipeline(steps=[("pca", PCA(n_components=10)), ("m", LogisticRegression())])
evaluate("PCA", model)
# No performance improvement here...
# Accuracy: 0.824 (0.034)

# Singular Value Decomposition (SVD) is popular for sparse data (many zero values)
model = Pipeline(
    steps=[("svd", TruncatedSVD(n_components=10)), ("m", LogisticRegression())]
)
evaluate("SVD", model)
# Still no improvement:
# Accuracy: 0.824 (0.034)

# Linear Discriminant Analysis (LDA) is multi-class classification algo where dimensions
# are limited to 1 and C-1 where C is number of classes
# In our example we use a binary classification so we limit the number of dimensions to 1
model = Pipeline(
    steps=[
        ("lda", LinearDiscriminantAnalysis(n_components=1)),
        ("m", LogisticRegression()),
    ]
)
evaluate("LDA", model)
# Got a _little_ improvement:
# Accuracy: 0.825 (0.034)

# Isomap Embedding creates an embedding of the dataset and attempts to preserve the
# relationships in the dataset
model = Pipeline(
    steps=[
        ("iso", Isomap(n_components=10)),
        ("m", LogisticRegression()),
    ]
)
evaluate("Isomap", model)
# Got some more improvement:
# Accuracy Mean: 0.888    Accuracy STD: (0.029)

# Local linear embedding (LLE) creates an embedding of the dataset and attempts to
# preserve the relationships between neighbors in the dataset
model = Pipeline(
    steps=[
        ("lle", LocallyLinearEmbedding(n_components=10)),
        ("m", LogisticRegression()),
    ]
)
evaluate("LLE", model)
# Pretty good improvement:
# Accuracy Mean: 0.886    Accuracy STD: (0.028)

# Modified Locally Linear Embedding (Modified LLE) is an extension of LLE that creates
# multiple weighting vectors for each neighborhood
model = Pipeline(
    steps=[
        (
            "modified-lle",
            LocallyLinearEmbedding(n_components=5, method="modified", n_neighbors=10),
        ),
        ("m", LogisticRegression()),
    ]
)
evaluate("Modified LLE", model)
# Not as good as base LLE:
# Modified LLE: Accuracy Mean: 0.848      Accuracy STD: (0.038)
