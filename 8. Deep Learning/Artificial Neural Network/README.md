# Artificial Neural Network

인공 신경망(Artificial Neural Network)이 발전함에 따라, 앞서 배운 퍼셉트론의 틀을 벗어나 계단 함수(Step funtion)가 아닌 다른 활성화 함수가 도입되고, 은닉층의 구조를 좀 더 복잡하게 해보는 등 다양한 시도로 새로운 인공 신경망들이 등장하기 시작했다. 이번 챕터에서는 계단 함수 대신 선택할 수 있는 활성화 함수들과 인공 신경망의 행렬 연산을 이용한 순전파에 대해

## 1. 피드 포워드 신경망(Feed-Forward Neural Network, FFNN)

단층 퍼셉트론이나 다층 퍼셉트론과 같이 입력층에서 출력층 방향으로 연산이 전개되는 신경망

## 2. 전결합층(Fully-connected layer, FC, Dense layer)

다층 퍼셉트론은 은닉층과 출력층에 있는 모든 뉴런은 바로 이전 층의 모든 뉴런과 연결돼 있었다. 그와 같이 어떤 층의 모든 뉴런이 이전 층의 모든 뉴런과 연결돼 있는 층을 전결합층이라 한다. 이와 동일한 의미로 밀집층(Dense layer)이라고 하기도 한다.

## 3. 활성화 함수(Activation Function)

은닉층과 출력층의 뉴런에서 출력값을 결정하는 함수

### 활성화 함수의 특징 - 비선형 함수(Nonlinear function)

선형 함수를 사용하게 되면 은닉층을 쌓을 수가 없고 선형 함수로는 은닉층을 여러번 추가하더라도 1회 추가한 것과 차이를 줄 수 없다.

### 계단 함수(Step function)

계단 함수는 이제 거의 사용되지 않지만, 퍼셉트론을 통해 처음으로 인공 신경망을 배울 때 가장 처음 접하게 되는 활성화 함수

### 시그모이드 함수(Sigmoid function)

시그모이드 함수는 가중치 곱의 합을 0과 1사이의 값으로 조정하여 반환하는 활성화 함수, 이진 분류 문제(Binary Classification)에 사용

### 렐루 함수(Relu function)

은닉층에서 활성화 함수로 가장 많이 사용되는 활성화 함수, 하지만 입력값이 0보다 작을 경우, 미분값이 0이 되는 단점이 존재하는데 이를 보완한 Leaky ReLU와 같은 ReLU의 변형 함수들이 등장하기 시작

### 하이퍼볼릭탄젠트 함수(Tanh function)

은닉층에서 활성화 함수로 종종 사용되는 활성화 함수, 이미지 인식 분야에서 자주 사용되는 인공 신경망인 CNN에서는 ReLu 함수가 주로 사용되고, 자연어 처리 분야에서 자주 사용되는 인공 신경망인 LSTM에서는 tanh 함수와 시그모이드 함수가 주로 사용

### 소프트맥스 함수(Softmax function)

세 가지 이상의 선택지 중 하나를 고르는 다중 클래스 분류(MultiClass Classification) 문제에 주로 사용

ex) test1.py

## 4. 행렬의 곱셈을 이용한 순전파(Forward Propagation)

인공 신경망에서 입력층에서 출력층 방향으로 연산을 진행하는 과정을 순전파(Forward Propagation)라고 한다.

이때 인공 신경망은 순전파와는 반대 방향으로 연산을 진행하며 가중치를 업데이트하는데, 이 과정을 역전파(BackPropagation)라고 한다.

입력층 : 4개의 입력과 8개의 출력
은닉층1 : 8개의 입력과 8개의 출력
은닉층2 : 8개의 입력과 3개의 출력
출력층 : 3개의 입력과 3개의 출력

layer 1 : X<sub>1 × 4</sub> × W<sub>4 × 8</sub> + B<sub>1 × 8</sub> = Y<sub>1 × 8</sub>
layer 2 : X<sub>1 × 8</sub> × W<sub>8 × 8</sub> + B<sub>1 × 8</sub> = Y<sub>1 × 8</sub>
layer 3 : X<sub>1 × 8</sub> × W<sub>8 × 3</sub> + B<sub>1 × 3</sub> = Y<sub>1 × 3</sub>

ex) test2.py
