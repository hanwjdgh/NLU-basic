import nltk
from nltk.stem import PorterStemmer

s = PorterStemmer()
words=['formalize', 'allowance', 'electricical']
lst = [s.stem(w) for w in words]

print(lst)

"""
-> ['formal', 'allow', 'electric']
"""