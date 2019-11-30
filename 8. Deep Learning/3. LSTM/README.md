# LSTM

RNN은 비교적 짧은 시퀀스(sequence)에 대해서만 효과를 보이는 단점이 있습니다. 즉, RNN의 time-step이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생 (장기 의존성 문제(the problem of Long-Term Dependencies))

이러한 단점을 보완한 RNN의 일종을 장단기 메모리(Long Short-Term Memory)라고 하며, 줄여서 LSTM

## 1. 단층-단방향 & many-to-one

x = {1,2,3,4,5}, y = {6}
z = {2,3,4,5,6} → {7}

ex) test1.py

## 2. 단층-단방향 & many-to-many

x = {1,2,3,4,5}, y = {2,3,4,5,6}

ex) test2.py, test3.py, test4.py

test2 = many-to-many
test3 = many-to-many wrong case
test4 = many-to-many → many-to-one

## 3. 단층-양방향 & many-to-one/ many-to-many

양방향 (bidirectional) LSTM은 순차적인 입력값에 대해 이전 데이터와의 관계뿐만 아니라 이후 데이터와의 관계까지도 학습

x = {1,2,3,4,5} 일때, 3번 째 스텝에서는 Forward와 Backward 레이어에 각각 3이 입력된다.

Forward = 1 → 2 → 3
Backward = 3 ← 4 ← 5

ex) test5.py, test6.py

test5 = many-to-one
test6 = many-to-many

## 4. 2층-단방향/양방향 & many-to-one

return_sequences=True, 1층의 모든 스텝에서의 출력이 2층의 각 스텝으로 전달돼야 하기 때문에 1층에는 이 옵션을 사용해야 한다.

ex) test7.py, test8.py

test7.py = many-to-one
test8.py = many-to-many
