from nltk import word_tokenize, pos_tag, ne_chunk

sentence = "James is working at Disney in London"
sentence=pos_tag(word_tokenize(sentence))

print(sentence) # 토큰화와 품사 태깅을 동시 수행

"""
→
[('James', 'NNP'), ('is', 'VBZ'), ('working', 'VBG'), ('at', 'IN'), ('Disney', 'NNP'), ('in', 'IN'), ('London', 'NNP')]
"""

sentence=ne_chunk(sentence)
print(sentence) # 개체명 인식

"""
→
(S
  (PERSON James/NNP)
  is/VBZ
  working/VBG
  at/IN
  (ORGANIZATION Disney/NNP)
  in/IN
  (GPE London/NNP))
"""
