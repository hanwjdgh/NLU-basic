import numpy as np
from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트

X=np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# 입력 벡터의 차원은 2입니다. 즉, input_dim은 2입니다.
y=np.array([0, 1, 1, 1])
# 출력 벡터의 차원은 1입니다. 즉, output_dim은 1입니다.

model=Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid')) # 이제 입력의 차원은 2입니다.

model.compile(optimizer='sgd' ,loss='binary_crossentropy',metrics=['binary_accuracy'])
# 옵티마이저는 경사하강법의 변형인 확률적 경사 하강법 sgd를 사용합니다.
# 손실 함수(Loss function)는 binary_crossentropy(이진 크로스 엔트로피)를 사용합니다.

model.fit(X,y, batch_size=1, epochs=800, shuffle=False)
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 800번 시도합니다.

print(model.predict(X))

"""
→
[[0.45521256]
 [0.84107596]
 [0.8577089 ]
 [0.97447586]]
"""