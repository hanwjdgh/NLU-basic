from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten

sentences = ['멋있어 최고야 짱이야 감탄이다', '헛소리 지껄이네', '닥쳐 자식아', '우와 대단하다', '우수한 성적', '형편없다', '최상의 퀄리티']
y_train = [1, 0, 0, 1, 1, 0, 1]

t = Tokenizer()
t.fit_on_texts(sentences)
vocab_size = len(t.word_index) + 1

print(vocab_size)

"""
→ 16
"""

X_encoded = t.texts_to_sequences(sentences)
print(X_encoded)

"""
→ [[1, 2, 3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13], [14, 15]]
"""

max_len=max(len(l) for l in X_encoded)
print(max_len)

"""
→ 4
"""

X_train=pad_sequences(X_encoded, maxlen=max_len, padding='post')
print(X_train)

"""
→ 
[[ 1  2  3  4]
 [ 5  6  0  0]
 [ 7  8  0  0]
 [ 9 10  0  0]
 [11 12  0  0]
 [13  0  0  0]
 [14 15  0  0]]
"""

model = Sequential()
model.add(Embedding(vocab_size, 4, input_length=max_len)) # 모든 임베딩 벡터는 4차원을 가지게됨.
model.add(Flatten()) # Dense의 입력으로 넣기위함임.
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X_train, y_train, epochs=100, verbose=2)