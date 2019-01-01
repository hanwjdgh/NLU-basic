#import nltk
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

#nltk.download('stopwords')
text=["Family is not an important thing. It's everything."]
sw = stopwords.words("english")
vect = CountVectorizer(stop_words =sw)

print(vect.fit_transform(text).toarray()) 

"""
-> [[1 1 1 1]]
"""

print(vect.vocabulary_)

"""
-> {'family': 1, 'important': 2, 'thing': 3, 'everything': 0}
"""