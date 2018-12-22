import nltk
from nltk.stem import WordNetLemmatizer

# 표제어 추출을 위한 도구 = WordNetLemmatizer

n=WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
lst = [n.lemmatize(w) for w in words]

print(lst)

"""
-> ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']
"""