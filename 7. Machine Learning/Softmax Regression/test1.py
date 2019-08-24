import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트

data = pd.read_csv('iris.csv',encoding='latin1')

print(len(data))
print(data[:5])
print("품종 종류:", data["Species"].unique(), sep="\n")

"""
→
150
   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa
품종 종류:
['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']
"""
data['Species'] = data['Species'].replace(['Iris-virginica','Iris-setosa','Iris-versicolor'],[0,1,2])
data_X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values # X 데이터에 해당됩니다. X는 총 4개입니다.
data_y = data['Species'].values # Y 데이터에 해당됩니다. 예측해야하는 값입니다.

print(data_X[:5]) #X에 해당되는 데이터를 5개만 출력합니다.
print(data_y[:5]) #y에 해당되는 데이터를 5개만 출력합니다.

"""
→
[[5.1 3.5 1.4 0.2]
 [4.9 3.  1.4 0.2]
 [4.7 3.2 1.3 0.2]
 [4.6 3.1 1.5 0.2]
 [5.  3.6 1.4 0.2]]
[1 1 1 1 1]
"""

(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size=0.8, random_state=1)
# 훈련 데이터와 테스트 데이터를 8:2로 나눕니다. 또한 데이터의 순서를 섞습니다.

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# 훈련 데이터와 테스트 데이터에 대해서 원-핫 인코딩

print(y_train[:5])
print(y_test[:5])

"""
→
[[0. 0. 1.]
 [1. 0. 0.]
 [0. 0. 1.]
 [1. 0. 0.]
 [1. 0. 0.]]
[[0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [0. 1. 0.]
 [1. 0. 0.]]
"""

model=Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
sgd=optimizers.SGD(lr=0.01)
# 학습률(learning rate, lr)은 0.01로 합니다.

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# 옵티마이저는 경사하강법의 일종인 adam을 사용합니다.
# 손실 함수(Loss function)은 평균제곱오차 크로스 엔트로피 함수를 사용합니다.

history=model.fit(X_train,y_train, batch_size=1, epochs=200, validation_data=(X_test, y_test))
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 200번 시도합니다.

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

"""
→ 테스트 정확도: 0.9667
"""