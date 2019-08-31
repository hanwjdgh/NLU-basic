from keras.datasets import reuters
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils

(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=1000, test_split=0.2)    

max_len=100
X_train = sequence.pad_sequences(X_train, maxlen=max_len) # 훈련용 뉴스 기사 패딩
X_test = sequence.pad_sequences(X_test, maxlen=max_len) # 테스트용 뉴스 기사 패딩

y_train = np_utils.to_categorical(y_train) # 훈련용 뉴스 기사 레이블의 원-핫 인코딩
y_test = np_utils.to_categorical(y_test) # 테스트용 뉴스 기사 레이블의 원-핫 인코딩

model = Sequential()
model.add(Embedding(1000, 120))
model.add(LSTM(120))
model.add(Dense(46, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=100, epochs=20, validation_data=(X_test,y_test))

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

""" 
→ 테스트 정확도: 0.7137
"""