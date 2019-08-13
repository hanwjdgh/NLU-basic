import nltk
from nltk.tokenize import RegexpTokenizer

# 정규 표현식을 사용한 단어 토큰화를 수행 = RegexpTokenizer

tokenizer=RegexpTokenizer("[\w]+")
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))

"""
→ ['Don', 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'Mr', 'Jone', 's', 'Orphanage', 'is', 'as', 'cheery', 
   'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop'] 
   문자 또는 숫자가 한 개 이상 존재하는 경우
"""

# gaps=true는 해당 정규 표현식을 토큰으로 나누기 위한 기준으로 사용
tokenizer=RegexpTokenizer("[\s]+", gaps=True)
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))

"""
→ ["Don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name,', 'Mr.', "Jone's", 'Orphanage', 'is', 'as', 'cheery', 'as', 
   'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
"""

# gaps=True라는 부분을 기재하지 않는다면, 토큰화의 결과는 공백들
tokenizer=RegexpTokenizer("[\s]+")
print(tokenizer.tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop"))

"""
→ [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']
"""