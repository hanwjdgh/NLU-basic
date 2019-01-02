from sklearn.feature_extraction.text import CountVectorizer

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])

print(vect.fit_transform(text).toarray()) 

"""
-> [[1 1 1 1 1]]
"""

print(vect.vocabulary_)

"""
-> {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}
"""