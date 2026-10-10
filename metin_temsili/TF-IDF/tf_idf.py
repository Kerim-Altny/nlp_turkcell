from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

documents = [
    "cat likes milk",
    "cat likes fish",
    "dog likes fish",
    "dog eats meat"
]
tfidf_vectorizer = TfidfVectorizer()
X = tfidf_vectorizer.fit_transform(documents)
feature_names = tfidf_vectorizer.get_feature_names_out()

print("Feature Names:", feature_names)
print("TF-IDF Matrix:\n", X.toarray())