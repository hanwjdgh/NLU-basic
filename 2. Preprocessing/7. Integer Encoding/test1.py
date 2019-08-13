from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

text="""A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. 
     Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. 
     But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."""

vocab = Counter()

sentences = []
stop_words = set(stopwords.words('english'))

for i in text:
    sentence = word_tokenize(i)
    result = []

    for word in sentence:
        word = word.lower()
        if word not in stop_words:
            if len(word) > 2:
                result.append(word)
                vocab[word] = vocab[word]+1
    sentences.append(result)
print(sentences)

"""
→ 
[['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], 
 ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], 
 ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]
"""

print(vocab)

"""
→ 
Counter({'barber': 8,
         'person': 3,
         'good': 1,
         'huge': 5,
         'knew': 1,
         'secret': 6,
         'kept': 4,
         'word': 2,
         'keeping': 2,
         'driving': 1,
         'crazy': 1,
         'went': 1,
         'mountain': 1})
"""

vocab_sorted=sorted(vocab.items(), key=lambda x:x[1], reverse=True)
print(vocab_sorted)

"""
→ 
[('barber', 8), ('secret', 6), ('huge', 5), ('kept', 4), ('person', 3), ('word', 2), ('keeping', 2), ('good', 1), ('knew', 1), 
 ('driving', 1), ('crazy', 1), ('went', 1), ('mountain', 1)]
"""

word_to_index={}
i=0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # Cleaning
        i=i+1
        word_to_index[word]=i
print(word_to_index)

"""
→ 
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5, 'word': 6, 'keeping': 7}
"""