import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")

# Stemming & lemmatization are techniques to normalize text
# reading -> read
# books -> book

# Stemming
stemmer = PorterStemmer()

phrase = "reading the books"
words = word_tokenize(phrase)

stemmed_words = []
for word in words:
    stemmed_words.append(stemmer.stem(word))

print(" ".join(stemmed_words))

# Lemmatizing
lemmatizer = WordNetLemmatizer()

lemmatized_words = []
for word in words:
    lemmatized_words.append(lemmatizer.lemmatize(word, pos="v"))

print(" ".join(lemmatized_words))
