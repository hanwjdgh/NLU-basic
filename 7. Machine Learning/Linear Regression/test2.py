from keras.models import Sequential # 케라스의 Sequential()을 임포트
from keras.layers import Dense # 케라스의 Dense()를 임포트
from keras import optimizers # 케라스의 옵티마이저를 임포트
import numpy as np # Numpy를 임포트

X=np.array([[70,85,11],[71,89,18],[50,80,20],[99,20,10],[50,10,10]]) # 중간, 기말, 가산점
# 입력 벡터의 차원은 3입니다. 즉, input_dim은 3입니다.
y=np.array([73,82,72,57,34]) # 최종 성적
# 출력 벡터의 차원은 1입니다. 즉, output_dim은 1입니다.

model=Sequential()
model.add(Dense(1, input_dim=3, activation='linear'))
sgd=optimizers.SGD(lr=0.00001)
# 학습률(learning rate, lr)은 0.01로 합니다.

model.compile(optimizer=sgd ,loss='mse',metrics=['mse'])
# 옵티마이저는 경사하강법의 변형인 확률적 경사 하강법 sgd를 사용합니다.
# 손실 함수(Loss function)은 평균제곱오차 mse를 사용합니다.

model.fit(X,y, batch_size=1, epochs=2000, shuffle=False)
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 2,000번 시도합니다.

print(model.predict(X))

"""
→
[[73.15294 ]
 [81.98001 ]
 [71.93192 ]
 [57.161617]
 [33.669353]]
"""

X_test=np.array([[20,99,10],[40,50,20]]) # 각각 58점과 56점을 예측해야 합니다.
print(model.predict(X_test))

"""
→
[[58.08134 ]
 [55.734634]]
"""