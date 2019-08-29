# Keras

케라스는 유저가 손쉽게 딥 러닝을 구현할 수 있도록 도와주는 상위 레벨의 인터페이스, 케라스를 사용하면 딥 러닝을 쉽게 구현할 수 있다.

## 1. 전처리(Preprocessing)

Tokenizer() : 토큰화와 정수 인코딩

pad_sequence() : 샘플의 길이를 동일하게 맞춰 주는 함수

pad_sequence()의 인자

- 첫번째 인자 : 패딩을 진행할 데이터
- maxlen : 모든 데이터에 대해서 정규화 할 길이
- padding : 'pre'를 선택하면 앞에 0을 채우고 'post'를 선택하면 뒤에 0을 채움.

ex) test1.py

## 2. 워드 임베딩(Word Embedding)

워드 임베딩이란 텍스트 내의 단어들을 밀집 벡터(dense vector)로 만드는 것

|-|원-핫 벡터 | 임베딩 벡터 |
|:--------|:--------:|:--------:|
|차원|고차원(단어 집합의 크기)|저차원|
|다른 표현|희소|벡터의 일종|밀집 벡터의 일종|
|표현 방법|수동|훈련 데이터로부터 학습함|
|값의 타입|1과 0|실수|

원-핫 벡터의 차원이 주로 20,000 이상을 넘어가는 것과는 달리 임베딩 벡터는 주로 256, 512, 1024 등의 차원을 가진다. 임베딩 벡터는 초기에는 랜덤값을 가지지만, 인공 신경망의 가중치가 학습되는 방법과 같은 방식으로 값이 학습되며 변경된다.

Embedding() : 단어를 밀집 벡터로 만드는 역할, 2D → 3D

Embedding()의 인자

- 첫번째 인자 : 단어 집합의 크기. 즉, 총 단어의 개수
- 두번째 인자 : 임베딩 벡터의 출력 차원. 결과로서 나오는 임베딩 벡터의 크기
- input_length : 입력 시퀀스의 길이

ex) Embedding(7, 2, input_length=5)

## 3. 모델링(Modeling)

Sequential() : 층을 구성하기 위해 사용

```
from keras.models import Sequential
model = Sequential()
model.add(...) # 층 추가
model.add(...) # 층 추가
model.add(...) # 층 추가

model.add(Dense(1, input_dim=3, init='uniform', activation='relu'))
```

Dense() : 전결합층(fully-conntected layer)을 추가), summary() (모델의 정보를 요약해서 보여줌

Dense()의 인자

- 첫번째 인자 : 출력 뉴런의 수
- input_dim : 입력 뉴런의 수 (입력의 차원)
- init : 가중치 초기화 방법
    - uniform : 균일 분포
    - normal : 가우시안 분포
- activation : 활성화 함수
    - linear : 디폴트 값으로 별도 활성화 함수 없이 입력 뉴런과 가중치의 계산 결과 그대로 출력
    - sigmoid : 시그모이드 함수. 이진 분류 문제에서 출력층에 주로 사용되는 활성화 함수
    - softmax : 소프트맥스 함수. 셋 이상을 분류하는 다중 클래스 분류 문제에서 출력층에 주로 사용되는 활성화 함수
    - relu : 렐루 함수. 은닉층에 주로 사용되는 활성화 함수

## 4. 컴파일(Compile)과 훈련(Training)

compile()

```
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
```

compile()의 인자

- optimizer : 훈련 과정을 설정하는 옵티마이저를 설정, 'adam'이나 'sgd'와 같이 문자열로 지정할 수도 있습니다.
- loss : 훈련 과정에서 사용할 손실 함수(loss function)를 설정
- metrics : 훈련을 모니터링하기 위한 지표를 선택

대표적인 조합

|문제 유형|손실 함수명|출력층의 활성화 함수명|
|:--------|:--------:|:--------:|
|회귀 문제|mean_squared_error(평균 제곱 오차)|-|
다중 클래스 분류|categorical_crossentropy (범주형 교차 엔트로피)|소프트맥스|
|다중 클래스 분류|sparse_categorical_crossentropy|소프트맥스|
|이진 분류|binary_crossentropy(이항 교차 엔트로피)|시그모이드|

fit() : 모델을 학습

```
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_data(X_val, y_val))
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0, validation_split=0.2))
```

fit()의 인자

- 첫번째 인자 : 훈련 데이터
- 두번째 인자 : 지도 학습 관점에서 레이블 데이터
- epochs : 에포크라고 읽으며 에포크 1은 전체 데이터를 한 차례 훑고 지나갔음을 의미, 정수값 기재 필요, 총 훈련 횟수를 정의
- batch_size : 배치 크기. 기본값은 32. 미니 배치 경사 하강법을 사용하고 싶지 않을 경우에는 batch_size=None을 통해 선택 가능
- verbose = 학습 중 출력되는 문구를 설정
    - 0 : 아무 것도 출력하지 않습니다.
    - 2 : 에포크 횟수당 한 줄씩 출력합니다.
    - 1 : 훈련의 진행도를 보여주는 진행 막대를 보여줍니다.
- validation_data(x_val, y_val) : 검증 데이터(validation data)를 사용합니다. 검증 데이터를 사용하면 각 에포크마다 검증 데이터의 정확도도 함께 출력되는데, 이 정확도는 훈련이 잘 되고 있는지를 보여줄 뿐이며 실제로 모델이 검증 데이터를 학습하지는 않습니다. 검증 데이터의 loss가 낮아지다가 높아지기 시작하면 이는 과적합(overfitting)의 신호입니다.
- validation_split : validation_data 대신 사용할 수 있습니다. 검증 데이터를 사용하는 것은 동일하지만, 별도로 존재하는 검증 데이터를 주는 것이 아니라 X_train과 y_train에서 일정 비율을 분리하여 이를 검증 데이터로 사용합니다. 역시나 훈련 자체에는 반영되지 않고 훈련 과정을 지켜보기 위한 용도로 사용됩니다. 아래는 validation_data 대신에 validation_split을 사용했을 경우를 보여줍니다.

## 5. 평가(Evaluation)와 예측(Prediction)

evaluate() : 테스트 데이터를 통해 학습한 모델에 대한 정확도를 평가

```
model.evaluate(X_test, y_test, batch_size=32)
```

evaluate()의 인자

- 첫번째 인자 : 테스트 데이터
- 두번째 인자 : 지도 학습 관점에서 레이블 테스트 데이터
- batch_size : 배치 크기

predict() : 임의의 입력에 대한 모델의 출력값을 확인

```
model.predict(X_input, batch_size=32)
```

predict()의 인자

- 첫번째 인자 : 예측하고자 하는 데이터
- batch_size : 배치 크기

## 6. 모델의 저장(Save)과 로드(Load)

save() : 인공 신경망 모델을 hdf5 파일에 저장

load_model() : 저장해둔 모델을 불러온다.

```
from keras.models import load_model

model.save("model_name.h5")
model = load_model("model_name.h5")
```

## 7. 케라스의 함수형 API(Keras Functional API)

### 1. sequential API로 만든 모델

ex) test2.py

### 2. functional API로 만든 모델

sequential API와는 다르게 functional API에서는 입력 데이터의 크기(shape)를 인자로 입력층을 정의

ex) test3.py
