import pandas as pd
from string import punctuation
from keras_preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.utils import to_categorical
from keras.layers import Embedding, Dense, LSTM
from keras.models import Sequential

df=pd.read_csv('ArticlesApril2018.csv')

print(df.columns)
print('열의 개수: ',len(df.columns))

"""
→
Index(['articleID', 'articleWordCount', 'byline', 'documentType', 'headline',
       'keywords', 'multimedia', 'newDesk', 'printPage', 'pubDate',
       'sectionName', 'snippet', 'source', 'typeOfMaterial', 'webURL'],
      dtype='object')
열의 개수:  15
"""

print(df['headline'].isnull().values.any())

"""
→ False
"""

headline = [] # 리스트 선언
headline.extend(list(df.headline.values)) # 헤드라인의 값들을 리스트로 저장
print(headline[:5]) # 상위 5개만 출력

"""
→
['Former N.F.L. Cheerleaders’ Settlement Offer: $1 and a Meeting With Goodell', 'E.P.A. to Unveil a New Rule. Its Effect: Less Science in Policymaking.', 
 'The New Noma, Explained', 'Unknown', 'Unknown']
"""

print(len(headline))

"""
→ 1324
"""

headline = [n for n in headline if n != "Unknown"] # Unknown 값을 가진 샘플 제거
print(len(headline))
print(headline[:5])

"""
→
1214
['Former N.F.L. Cheerleaders’ Settlement Offer: $1 and a Meeting With Goodell', 'E.P.A. to Unveil a New Rule. Its Effect: Less Science in Policymaking.', 'The New Noma, Explained', 
 'How a Bag of Texas Dirt  Became a Times Tradition', 'Is School a Place for Self-Expression?']
"""

def repreprocessing(s):
    s=s.encode("utf8").decode("ascii",'ignore')
    return ''.join(c for c in s if c not in punctuation).lower() # 구두점 제거와 동시에 소문자화

text = [repreprocessing(x) for x in headline]
print(text[:5])

"""
→
['former nfl cheerleaders settlement offer 1 and a meeting with goodell', 'epa to unveil a new rule its effect less science in policymaking',
 'the new noma explained', 'how a bag of texas dirt  became a times tradition', 'is school a place for selfexpression']
"""

t = Tokenizer()
t.fit_on_texts(text)
vocab_size = len(t.word_index) + 1
print('단어 집합의 크기 : %d' % vocab_size)

"""
→ 단어 집합의 크기 : 3494
"""

sequences = list()

for line in text: # 1,214 개의 샘플에 대해서 샘플을 1개씩 가져온다.
    encoded = t.texts_to_sequences([line])[0] # 각 샘플에 대한 정수 인코딩
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

print(sequences[:11])

"""
→
[[99, 269], [99, 269, 371], [99, 269, 371, 1115], [99, 269, 371, 1115, 582], [99, 269, 371, 1115, 582, 52], 
 [99, 269, 371, 1115, 582, 52, 7], [99, 269, 371, 1115, 582, 52, 7, 2], [99, 269, 371, 1115, 582, 52, 7, 2, 372], 
 [99, 269, 371, 1115, 582, 52, 7, 2, 372, 10], [99, 269, 371, 1115, 582, 52, 7, 2, 372, 10, 1116], [100, 3]]
"""

index_to_word={}
for key, value in t.word_index.items(): # 인덱스를 단어로 바꾸기 위해 index_to_word를 생성
    index_to_word[value] = key
print(index_to_word[582])

"""
→ offer
"""

max_len=max(len(l) for l in sequences)
print(max_len)

"""
→ 24
"""

sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(sequences[:3])

"""
→
[[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0    0   99  269]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0   99  269  371]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   99  269  371 1115]
"""

sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]
print(X[:3])
print(y[:3])

"""
→
[[ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0    0   99]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0    0   99  269]
 [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0   99  269  371]
[ 269  371 1115]
"""

y = to_categorical(y, num_classes=vocab_size)

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_len-1))
# y데이터를 분리하였으므로 이제 X데이터의 길이는 기존 데이터의 길이 - 1
model.add(LSTM(128))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, verbose=2)

def sentence_generation(model, t, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n): # n번 반복
        encoded = t.texts_to_sequences([current_word])[0] # 현재 단어에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=23, padding='pre') # 데이터에 대한 패딩
        result = model.predict_classes(encoded, verbose=0)
    # 입력한 X(현재 단어)에 대해서 y를 예측하고 y(예측한 단어)를 result에 저장.
        for word, index in t.word_index.items(): 
            if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break # 해당 단어가 예측 단어이므로 break
        current_word = current_word + ' '  + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word # 예측 단어를 문장에 저장
    # for문이므로 이 행동을 다시 반복
    sentence = init_word + sentence
    return sentence

print(sentence_generation(model, t, 'i', 10))
print(sentence_generation(model, t, 'how', 10))

"""
→
i disapprove of school vouchers can i still apply for them
how to make facebook more accountable will so your neighbor chasing
"""