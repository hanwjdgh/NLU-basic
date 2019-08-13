import nltk
from nltk.stem import WordNetLemmatizer

# 표제어 추출을 위한 도구 = WordNetLemmatizer

n=WordNetLemmatizer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
lst = [n.lemmatize(w) for w in words]

print(lst)

"""
→ ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']

표제어 추출은 단어의 형태가 적절히 보존된다. 하지만 'dy', 'ha'와 같이 의미를 알수없는 단어를 추출하는 경우도 있다.
이는 표제어 추출기가 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있기 때문이다.
"""

print(n.lemmatize('dies','v'))

"""
→ 'die'
"""

print(n.lemmatize('watched','v'))

"""
→ 'watch'
"""

print(n.lemmatize('has','v'))

"""
→ 'have'
"""