import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Lambda, Input

sess = tf.Session()
K.set_session(sess)
# 세션 초기화. 이는 텐서플로우 개념.

elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)
# 텐서플로우 허브로부터 ELMo를 다운로드

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

data = pd.read_csv('spam.csv 파일의 경로', encoding='latin-1')

data['v1'] = data['v1'].replace(['ham','spam'],[0,1])
y_data = list(data['v1'])
X_data = list(data['v2'])

n_of_train = int(len(X_data) * 0.8)
n_of_test = int(len(X_data) - n_of_train)

X_train = np.asarray(X_data[:n_of_train])
y_train = np.asarray(y_data[:n_of_train])
X_test = np.asarray(X_data[n_of_train:])
y_test = np.asarray(y_data[n_of_train:])

def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), as_dict=True, signature="default")["default"]
# 데이터의 이동이 케라스 → 텐서플로우 → 케라스가 되도록 하는 함수

input_text = Input(shape=(1,), dtype=tf.string)
embedding_layer = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
hidden_layer = Dense(256, activation='relu')(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)
model = Model(inputs=[input_text], outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=1, batch_size=60)  