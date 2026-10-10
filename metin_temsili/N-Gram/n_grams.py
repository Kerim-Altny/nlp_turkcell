from sklearn.feature_extraction.text import CountVectorizer

documents = [
    "I love programming in Python.",
    "Python is a great programming language.",
    "I enjoy solving problems with Python.",
    "Programming in Python is fun and rewarding."]

vectorizer_unigram = CountVectorizer(ngram_range=(1, 1))
vectorizer_bigram = CountVectorizer(ngram_range=(2, 2))
vectorizer_trigram = CountVectorizer(ngram_range=(3, 3))

X_unigram = vectorizer_unigram.fit_transform(documents)
X_bigram = vectorizer_bigram.fit_transform(documents)
X_trigram = vectorizer_trigram.fit_transform(documents)

unigram_features = vectorizer_unigram.get_feature_names_out()
bigram_features = vectorizer_bigram.get_feature_names_out()
trigram_features = vectorizer_trigram.get_feature_names_out()

print("Unigram Features:")
print(unigram_features)

print("\nBigram Features:")
print(bigram_features)

print("\nTrigram Features:")
print(trigram_features)