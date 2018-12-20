import nltk
from nltk.tokenize import TreebankWordTokenizer

# 구두점이나 특수 문자를 단순 제외하면 안되는 경우
# 줄임말과 단어 내에 띄어쓰기가 있는 경우 = TreebankWordTokenizer

tokenizer=TreebankWordTokenizer()
text="Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own."
print(tokenizer.tokenize(text))

"""
-> ['Starting', 'a', 'home-based', 'restaurant', 'may', 'be', 'an', 'ideal.', 'it', 'does', "n't", 'have', 'a', 
'food', 'chain', 'or', 'restaurant', 'of', 'their', 'own', '.']
"""