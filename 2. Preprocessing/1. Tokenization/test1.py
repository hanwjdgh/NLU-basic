import nltk  
from nltk.tokenize import word_tokenize  
from nltk.tokenize import WordPunctTokenizer

# 영어 토큰화 = word_tokenize, WordPunctTokenizer

print(word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
"""
→ ['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 
'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
"""


# WordPunctTokenizer은 구두점을 별도로 분류
print(WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
"""
→ ['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 
'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
"""