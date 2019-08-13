import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english')) 

word_tokens = word_tokenize(example)   
result = [] 

for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 

print(word_tokens) 
print(result) 

"""
→ 단어 토큰화 결과 : ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
  불용어 제거 결과 : ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
  'is', 'not', 'an' 제거
"""