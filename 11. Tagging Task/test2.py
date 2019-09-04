from collections import Counter
import re
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, TimeDistributed
from keras.optimizers import Adam
from keras.utils import np_utils
import numpy as np

vocab=Counter()

# https://raw.githubusercontent.com/Franck-Dernoncourt/NeuroNER/master/neuroner/data/conll2003/en/train.txt
f = open('train.txt', 'r')

sentences = []
sentence = []
ner_set = set()
# 파이썬의 set은 중복을 허용하지 않는다. 개체명 태깅의 경우의 수. 즉, 종류를 알아내기 위함이다.  

for line in f:
    if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
        if len(sentence) > 0:
            sentences.append(sentence)
            sentence=[]
        continue
    splits = line.split(' ')
    # 공백을 기준으로 속성을 구분한다.
    splits[-1] = re.sub(r'\n', '', splits[-1])
    # 개체명 태깅 뒤에 붙어있는 줄바꿈 표시 \n을 제거한다.
    word=splits[0].lower()
    # 단어들은 소문자로 바꿔서 저장한다. 단어의 수를 줄이기 위해서이다.
    vocab[word]=vocab[word]+1
    # 단어마다 빈도 수가 몇 인지 기록한다.
    sentence.append([word, splits[-1]])
    # 단어와 개체명 태깅만 기록한다.
    ner_set.add(splits[-1])
    # set에다가 개체명 태깅을 집어 넣는다. 중복은 허용되지 않으므로
    # 나중에 개체명 태깅이 어떤 종류가 있는지 확인할 수 있다.

vocab_sorted=sorted(vocab.items(), key=lambda x:x[1], reverse=True)

word_to_index = {w: i + 2 for i, (w, n) in enumerate(vocab_sorted) if n > 5}
word_to_index['PAD'] = 0  # 패딩을 위해 인덱스 0 할당
word_to_index['OOV'] = 1  # 모르는 단어을 위해 인덱스 1 할당
# 향후 훈련 데이터의 길이를 모두 맞추기위해 인덱스 0에는 'PAD'라는 단어를 넣고, 인덱스 1에는 단어 집합에 없는 단어들을 별도로 표시하기 위한 'OOV'라는 단어를 부여

ner_to_index={}
ner_to_index['PAD'] = 0
i=1
for ner in ner_set:
    ner_to_index[ner]=i
    i=i+1

# data_X를 만든다
data_X = []

for s in sentences:
    temp_X = []
    for w, label in s:
        try:
            temp_X.append(word_to_index.get(w,1))
        except KeyError:
            temp_X.append(word_to_index['OOV'])

    data_X.append(temp_X)

# data_y를 만든다
data_y = []

for s in sentences:
    temp_y = []
    for w, label in s:
            temp_y.append(ner_to_index.get(label))

    data_y.append(temp_y)

max_len=70
from keras.preprocessing.sequence import pad_sequences
pad_X = pad_sequences(data_X, padding='post', maxlen=max_len)
# data_X의 모든 샘플들의 길이를 맞출 때 뒤의 공간에 숫자 0으로 채움.
pad_y = pad_sequences(data_y, padding='post', maxlen=max_len)
# data_y의 모든 샘플들의 길이를 맞출 때 뒤의 공간에 숫자0으로 채움.

X_train, X_test, y_train, y_test = train_test_split(pad_X, pad_y, test_size=.2, random_state=777)

n_words = len(word_to_index)
n_labels = len(ner_to_index)
model = Sequential()
model.add(Embedding(input_dim=n_words, output_dim=16, input_length=max_len, mask_zero=True))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(TimeDistributed(Dense(n_labels, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer=Adam(0.001), metrics=['accuracy'])

y_train2 = np_utils.to_categorical(y_train)
y_train2[0][0]
array([0., 0., 1., 0., 0., 0., 0., 0., 0.], dtype=float32)
model.fit(X_train, y_train2, epochs=8)

y_test2 = np_utils.to_categorical(y_test)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test2)[1]))

index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value] = key

index_to_ner={}
for key, value in ner_to_index.items():
    index_to_ner[value] = key


i=10 # 확인하고 싶은 테스트용 샘플의 인덱스.
y_predicted = model.predict(np.array([X_test[i]])) # 입력한 테스트용 샘플에 대해서 예측 y를 리턴
y_predicted = np.argmax(y_predicted, axis=-1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.
true = np.argmax(y_test2[i], -1) # 원-핫 인코딩을 다시 정수 인코딩으로 변경함.

print("{:15}|{:5}|{}".format("단어", "실제값", "예측값"))
print(35 * "-")

for w, t, pred in zip(X_test[i], true, y_predicted[0]):
    if w != 0: # PAD값은 제외함.
        print("{:17}: {:7} {}".format(index_to_word[w], index_to_ner[t], index_to_ner[pred]))