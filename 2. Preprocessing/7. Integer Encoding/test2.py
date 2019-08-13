from keras.preprocessing.text import Tokenizer

text=["""A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret.
       Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a 
       huge secret to himself was driving the barber crazy. the barber went up a huge mountain."""]

t = Tokenizer()
t.fit_on_texts(text) # 텍스트의 리스트(list)를 가지고 단어 빈도수에 기반한 사전을 만들고 빈도수 순으로 단어에게 인덱스를 부여

print(t.word_index)

"""
→ 
{'a': 1, 'barber': 2, 'secret': 3, 'huge': 4, 'his': 5, 'is': 6, 'kept': 7, 'person': 8, 'the': 9, 'he': 10, 'word': 11, 'keeping': 12, 'good': 13, 
 'knew': 14, 'but': 15, 'and': 16, 'such': 17, 'to': 18, 'himself': 19, 'was': 20, 'driving': 21, 'crazy': 22, 'went': 23, 'up': 24, 
  'mountain': 25}
"""

print(t.word_counts) # 단어가 몇 개였는지 카운트 결과

"""
→ 
OrderedDict([('a', 8), ('barber', 8), ('is', 4), ('person', 3), ('good', 1), ('huge', 5), ('he', 2), ('knew', 1), ('secret', 6), ('the', 3), 
             ('kept', 4), ('his', 5), ('word', 2), ('but', 1), ('keeping', 2), ('and', 1), ('such', 1), ('to', 1), ('himself', 1), 
             ('was', 1), ('driving', 1), ('crazy', 1), ('went', 1), ('up', 1), ('mountain', 1)])
"""

print(t.texts_to_sequences(text)) # 입력으로 들어오는 코퍼스에 대하여 이미 정해진 인덱스로 변환

"""
→ 
[[1, 2, 6, 1, 8, 1, 2, 6, 13, 8, 1, 2, 6, 4, 8, 10, 14, 1, 3, 9, 3, 10, 7, 6, 4, 3, 4, 3, 5, 2, 7, 5, 11, 1, 2, 7, 5, 11, 5, 2, 7, 5, 3, 15, 
 12, 16, 12, 17, 1, 4, 3, 18, 19, 20, 21, 9, 2, 22, 9, 2, 23, 24, 1, 4, 25]]
"""

words_frequency = [w for w,c in t.word_counts.items() if c < 2] # 빈도수가 2미만 단어를 w라고 저장
for w in words_frequency:
    del t.word_index[w] # 해당 단어에 대한 인덱스 정보를 삭제
    del t.word_counts[w] # 해당 단어에 대한 카운트 정보를 삭제
print(t.texts_to_sequences(text))
print(t.word_index)

"""
→ 
[[1, 2, 6, 1, 8, 1, 2, 6, 8, 1, 2, 6, 4, 8, 10, 1, 3, 9, 3, 10, 7, 6, 4, 3, 4, 3, 5, 2, 7, 5, 11, 1, 2, 7, 5, 11, 5, 2, 7, 5, 3, 12, 12, 1, 4, 3, 9, 2, 9, 2, 1, 4]]
{'a': 1, 'barber': 2, 'secret': 3, 'huge': 4, 'his': 5, 'is': 6, 'kept': 7, 'person': 8, 'the': 9, 'he': 10, 'word': 11, 'keeping': 12}
"""