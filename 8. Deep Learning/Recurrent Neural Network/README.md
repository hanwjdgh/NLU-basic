# 순환 신경망(Recurrent Neural Network, RNN)

입력과 출력을 시퀀스 단위로 처리하는 시퀀스(Sequence) 모델

## 1. 순환 신경망(Recurrent Neural Network, RNN)

RNN은 은닉층의 노드에서 활성화 함수를 통해 나온 결과값을 출력층 방향으로도 보내면서, 다시 은닉층 노드의 다음 계산의 입력으로 보내는 특징을 갖고있다.

은닉층의 메모리 셀은 각각의 시점(time-step)에서 바로 이전 시점에서의 은닉층의 메모리 셀에서 나온 값을 자신의 입력으로 사용하는 재귀적 활동을 하고 있다.

- 일 대 다 : 하나의 사진 이미지 입력에 대해서 사진의 제목을 출력
- 다 대 일 : 입력 문서가 어떤 종류의 문서인지를 판별
- 다 대 다 : 입력 문장으로 부터 대답을 출력

RNN 층은 (batch_size, timesteps, input_dim) 크기의 3D 텐서를 입력으로 받는다.

SimpleRNN() : 케라스로 RNN 층을 추가

```
model.add(SimpleRNN(hidden_size)) 
model.add(SimpleRNN(hidden_size, input_shape=(timesteps, input_dim)))
model.add(SimpleRNN(hidden_size, input_length=M, input_dim=N))
# 단, M과 N은 정수
```

SimpleRNN()의 인자

- hidden_size : 은닉 상태의 크기를 정의. 메모리 셀이 다음 시점의 메모리 셀과 출력층으로 보내는 값의 크기(output_dim)와도 동일. RNN의 용량(capacity)을 늘린다고 보면 되며, 중소형 모델의 경우 보통 128, 256, 512, 1024 등의 값을 가진다.
- timesteps : 입력 시퀀스의 길이(input_length)
- input_dim : 입력의 크기

ex) test1.py, test2.py

## 2. BPTT(Backpropagation through time, BPTT)

RNN도 다른 인공 신경망과 마찬가지로 역전파를 통해서 학습을 진행, 피드 포워드 신경망의 역전파와 다른 점이 있다면, RNN은 전체 시점에 대해서 네트워크를 펼친 다음에 역전파를 사용하며 모든 시점에 대해서 가중치를 공유하고 있다는 점

## 3. 양방향 순환 신경망(Bidirectional Recurrent Neural Network)

양방향 순환 신경망은 시점 t에서의 출력값을 예측할 때 이전 시점의 데이터뿐만 아니라, 이후 데이터로도 예측할 수 있다는 아이디어에 기반

양방향 RNN은 하나의 출력값을 예측하기 위해 기본적으로 두 개의 메모리 셀을 사용, 첫번째 메모리 셀은 앞에서 배운 것처럼 앞 시점의 은닉 상태(Forward States)를 전달받아 현재의 은닉 상태를 계산하고 두번째 메모리 셀은 뒤 시점의 은닉 상태(Backward States)를 전달 받아 현재의 은닉 상태를 계산한다. 그리고 이 두 개의 값 모두가 하나의 출력값을 예측하기 위해 사용된다.

```
model = Sequential()
model.add(Bidirectional(SimpleRNN(hidden_size, return_sequences = True), input_shape=(timesteps, input_dim)))
```
