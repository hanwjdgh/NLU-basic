import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.layers import SimpleRNN, Embedding, Dense
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

data = pd.read_csv('spam.csv',encoding='latin1')

del data['Unnamed: 2']
del data['Unnamed: 3']
del data['Unnamed: 4']
data['v1'] = data['v1'].replace(['ham','spam'],[0,1])

print(data.isnull().values.any())

"""
→ False
"""

X_data = data['v2']
y_data = data['v1']
print(len(X_data))
print(len(y_data))

"""
→ 5572
  5572
"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data) #5572개의 행을 가진 X의 각 행에 토큰화를 수행
sequences = tokenizer.texts_to_sequences(X_data) #단어를 숫자값, 인덱스로 변환하여 저장

word_index = tokenizer.word_index

print((len(word_index)))

"""
→ 8920
"""

n_of_train = int(5572 * 0.8)
n_of_test = int(5572 - n_of_train)
print(n_of_train)
print(n_of_test)

"""
→ 4457
  1115
"""

X_data=sequences
print('메일의 최대 길이 :',max(len(l) for l in X_data))
print('메일의 평균 길이 :',sum(map(len, X_data))/len(X_data))

"""
→
메일의 최대 길이 : 189
메일의 평균 길이 : 15.794867193108399
"""

vocab_size = len(word_index)+1 # 단어의 수.
max_len = 189 # 전체 데이터셋의 길이는 189로 맞춥니다.
data = pad_sequences(X_data, maxlen=max_len)
print("data shape: ", data.shape)

""" 
→ data shape:  (5572, 189)
"""

X_test = data[n_of_train:] #X_data 데이터 중에서 뒤의 1115개의 데이터만 저장
y_test = y_data[n_of_train:] #y_data 데이터 중에서 뒤의 1115개의 데이터만 저장
X_train = data[:n_of_train] #X_data 데이터 중에서 앞의 4457개의 데이터만 저장
y_train = y_data[:n_of_train] #y_data 데이터 중에서 앞의 4457개의 데이터만 저장

model = Sequential()
model.add(Embedding(vocab_size, 32)) # 임베딩 벡터의 차원은 32
model.add(SimpleRNN(32)) # RNN 셀의 hidden_size는 32
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=4, batch_size=60, validation_split=0.2)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

"""
→ 테스트 정확도: 0.9749
"""
