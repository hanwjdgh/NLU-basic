import nltk
from nltk.tokenize import TreebankWordTokenizer

# 표준으로 쓰이고 있는 토큰화 방법 중 하나인 Penn Treebank Tokenization
"""
규칙 1. 하이푼으로 구성된 단어는 하나로 유지한다.
규칙 2. doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리해준다.
"""

tokenizer=TreebankWordTokenizer()

text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."

print(tokenizer.tokenize(text))

"""
→ ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', 
   "n't", 'have', 'a', 'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.'] 

결과를 보면, 각각 규칙1과 규칙2에 따라서 home-based는 하나의 토큰으로 취급하고 있으며, dosen't의 경우 does와 n't는 분리되었음을 볼 수 있습니다.
"""