import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Porter Algorithm

s = PorterStemmer()

text="This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."

# 단어 토큰화

words=word_tokenize(text)
print(words)

"""
→ ['This', 'was', 'not', 'the', 'map', 'we', 'found', 'in', 'Billy', 'Bones', "'s", 'chest', ',', 'but', 'an', 'accurate', 'copy', 
   ',', 'complete', 'in', 'all', 'things', '--', 'names', 'and', 'heights', 'and', 'soundings', '--', 'with', 'the', 'single', 
   'exception', 'of', 'the', 'red', 'crosses', 'and', 'the', 'written', 'notes', '.']
"""

lst = [s.stem(w) for w in words]

print(lst)

"""
→ ['thi', 'wa', 'not', 'the', 'map', 'we', 'found', 'in', 'billi', 'bone', "'s", 'chest', ',', 'but', 'an', 'accur', 'copi', 
   ',', 'complet', 'in', 'all', 'thing', '--', 'name', 'and', 'height', 'and', 'sound', '--', 'with', 'the', 'singl', 
   'except', 'of', 'the', 'red', 'cross', 'and', 'the', 'written', 'note', '.']
"""