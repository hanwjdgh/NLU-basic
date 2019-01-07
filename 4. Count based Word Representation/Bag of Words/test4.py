from sklearn.feature_extraction.text import CountVectorizer

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english")

print(vect.fit_transform(text).toarray())

"""
-> [[1 1 1]]
"""

print(vect.vocabulary_)

"""
-> {'family': 0, 'important': 1, 'thing': 2}
"""