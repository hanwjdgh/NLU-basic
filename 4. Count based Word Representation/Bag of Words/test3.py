#import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

#nltk.download('stopwords')

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])

print(vect.fit_transform(text).toarray()) 

"""
→ [[1 1 1 1 1]]
"""

print(vect.vocabulary_)

"""
→ {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}
"""

vect = CountVectorizer(stop_words="english")

print(vect.fit_transform(text).toarray())

"""
→ [[1 1 1]]
"""

print(vect.vocabulary_)

"""
→ {'family': 0, 'important': 1, 'thing': 2}
"""

sw = stopwords.words("english")
vect = CountVectorizer(stop_words =sw)

print(vect.fit_transform(text).toarray()) 

"""
→ [[1 1 1 1]]
"""

print(vect.vocabulary_)

"""
→ {'family': 1, 'important': 2, 'thing': 3, 'everything': 0}
"""