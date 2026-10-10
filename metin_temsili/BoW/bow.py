from sklearn.feature_extraction.text import CountVectorizer

sentences = [
    "ben elma seviyorum",
    "ben muz seviyorum",
    "elma çok güzel"
]

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(sentences)

print(vectorizer.get_feature_names_out())
print(X.toarray())