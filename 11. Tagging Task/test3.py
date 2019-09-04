import nltk
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding
from keras.optimizers import Adam
import numpy as np

tagged_sentences = nltk.corpus.treebank.tagged_sents() # 토큰화에 품사 태깅이 된 데이터 받아오기
# 미설치 에러 발생시 nltk.download('treebank')
print(tagged_sentences[0]) # 첫번째 문장 샘플 출력
print("품사 태깅이 된 문장 개수: ", len(tagged_sentences)) # 문장 샘플의 개수 출력

sentences, pos_tags =[], [] 
for tagged_sentence in tagged_sentences: # 3,914개의 문장 샘플을 1개씩 불러온다.
    sentence, tag_info = zip(*tagged_sentence) # 각 샘플에서 단어는 sentence에 품사 태깅 정보는 tags에 저장한다.
    sentences.append(np.array(sentence)) # 각 샘플에서 단어 정보만 저장한다.
    pos_tags.append(np.array(tag_info)) # 각 샘플에서 품사 태깅 정보만 저장한다.

vocab=Counter()
tag_set=set()

for sentence in sentences: # 훈련 데이터 X에서 문장 샘플을 1개씩 꺼내온다.
  for word in sentence: # 샘플에서 단어를 1개씩 꺼내온다.
    vocab[word.lower()]=vocab[word.lower()]+1 # 각 단어의 빈도수를 카운트한다.

for tags_list in pos_tags: # 훈련 데이터 y에서 품사 태깅 정보 샘플을 1개씩 꺼내온다.
  for tag in tags_list: # 샘플에서 품사 태깅 정보를 1개씩 꺼내온다.
    tag_set.add(tag) # 각 품사 태깅 정보에 대해서 중복을 허용하지 않고 집합을 만든다.

vocab_sorted=sorted(vocab.items(), key=lambda x:x[1], reverse=True)

word_to_index={'PAD' : 0, 'OOV' :1}
i=1
# 인덱스 0은 각각 입력값들의 길이를 맞추기 위한 PAD(padding을 의미)라는 단어에 사용된다.
# 인덱스 1은 모르는 단어를 의미하는 OOV라는 단어에 사용된다.
for (word, frequency) in vocab_sorted :
    # if frequency > 1 :
    # 빈도수가 1인 단어를 제거하는 것도 가능하겠지만 이번에는 별도 수행하지 않고 해보겠음.
    # 참고로 제거를 수행할 경우 단어 집합의 크기가 절반 정도로 줄어듬.
        i=i+1
        word_to_index[word]=i
print(word_to_index)
print(len(word_to_index))

tag_to_index={'PAD' : 0}
i=0
for tag in tag_set:
    i=i+1
    tag_to_index[tag]=i

data_X = []

for s in sentences:
    temp_X = []
    for w in s:
        try:
            temp_X.append(word_to_index.get(w.lower(),1))
        except KeyError: # 단어 집합을 만들 때 별도로 단어를 제거하지 않았기 때문에 이 과정에서는 OOV가 존재하지는 않음.
            temp_X.append(word_to_index['OOV'])

    data_X.append(temp_X)

data_y = []

for s in pos_tags:
    temp_y = []
    for w in s:
            temp_y.append(tag_to_index.get(w))

    data_y.append(temp_y)

max_len=150
from keras.preprocessing.sequence import pad_sequences
pad_X = pad_sequences(data_X, padding='post', maxlen=max_len)
# data_X의 모든 샘플의 길이를 맞출 때 뒤의 공간에 숫자 0으로 채움.
pad_y = pad_sequences(data_y, padding='post', value=tag_to_index['PAD'], maxlen=max_len)
# data_y의 모든 샘플의 길이를 맞출 때 뒤의 공간에 'PAD'에 해당되는 인덱스로 채움.
# 참고로 숫자 0으로 채우는 것과 'PAD'에 해당하는 인덱스로 채우는 것은 결국 0으로 채워지므로 같음

X_train, X_test, y_train, y_test = train_test_split(pad_X, pad_y, test_size=.2, random_state=777)
y_train2 = np_utils.to_categorical(y_train)

n_words = len(word_to_index)
n_labels = len(tag_to_index)

model = Sequential()
model.add(Embedding(n_words, 128, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(256, return_sequences=True)))
model.add(TimeDistributed(Dense(n_labels, activation=('softmax'))))
model.compile(loss='categorical_crossentropy',optimizer=Adam(0.001),metrics=['accuracy'])
model.fit(X_train, y_train2, batch_size=128, epochs=6)

y_test2 = np_utils.to_categorical(y_test)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test2)[1]))

index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value] = key

index_to_tag={}
for key, value in tag_to_index.items():
    index_to_tag[value] = key


i=10 # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]])) # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = np.argmax(y_predicted, axis=-1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.
true = np.argmax(y_test2[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for w, t, pred in zip(X_test[i], true, y_predicted[0]):
    if w != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[w], index_to_tag[t], index_to_tag[pred]))