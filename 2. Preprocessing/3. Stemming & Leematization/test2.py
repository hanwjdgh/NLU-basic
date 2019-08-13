import nltk
from nltk.stem import PorterStemmer

# Porter Algorithm
"""
ALIZE → AL
ANCE → 제거
ICAL → IC
"""

s = PorterStemmer()

words=['formalize', 'allowance', 'electricical']

lst = [s.stem(w) for w in words]
print(lst)

"""
→ ['formal', 'allow', 'electric']
"""