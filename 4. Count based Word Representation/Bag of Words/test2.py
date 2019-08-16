from sklearn.feature_extraction.text import CountVectorizer

# 단어의 빈도를 Count하여 Vector로 만드는 CountVectorizer

corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray())

"""
→ [[1 1 2 1 2 1]]
"""

print(vector.vocabulary_)

"""
→ {'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}
"""