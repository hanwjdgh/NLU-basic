import nltk
from nltk.stem import PorterStemmer

s=PorterStemmer()
words=['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
lst = [s.stem(w) for w in words]

print(lst)

"""
-> ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
"""

tst = [l.stem(w) for w in words]

print(tst)

"""
-> ['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']
"""