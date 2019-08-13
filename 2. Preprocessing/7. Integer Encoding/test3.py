from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

test=[8, 2, 5, 1, 3, 7, 9, 4, 6, 10]

for index, value in enumerate(test): 
    print("index : {}, value: {}".format(index,value))

"""
→ 
index : 0, value: 8
index : 1, value: 2
index : 2, value: 5
index : 3, value: 1
index : 4, value: 3
index : 5, value: 7
index : 6, value: 9
index : 7, value: 4
index : 8, value: 6
index : 9, value: 10
"""

text=[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], 
      ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], 
      ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

vocab=sum(text, [])
print(vocab)

"""
→ 
['barber', 'person', 'barber', 'good', 'person', 'barber', 'huge', 'person', 'knew', 'secret', 'secret', 'kept', 'huge', 'secret', 
 'huge', 'secret', 'barber', 'kept', 'word', 'barber', 'kept', 'word', 'barber', 'kept', 'secret', 'keeping', 'keeping', 'huge', 
 'secret', 'driving', 'barber', 'crazy', 'barber', 'went', 'huge', 'mountain']
"""

vocab=Counter() 
stop_words = set(stopwords.words('english'))

for word in sentence:
  word=word.lower()
  if word not in stop_words: 
    vocab[word]=vocab[word]+1 
print(vocab)

"""
→ 
Counter({'barber': 8, 'secret': 6, 'huge': 5, 'kept': 4, 'person': 3, 'word': 2, 'keeping': 2, 'good': 1, 'knew': 1, 'driving': 1, 
         'crazy': 1, 'went': 1, 'mountain': 1})
"""

word_to_index={word : index+1 for index, word in enumerate(vocab)}
print(word_to_index)

"""
→ 
{'barber': 1, 'person': 2, 'good': 3, 'huge': 4, 'knew': 5, 'secret': 6, 'kept': 7, 'word': 8, 'keeping': 9, 'driving': 10, 'crazy': 11, 
  'went': 12, 'mountain': 13}
"""