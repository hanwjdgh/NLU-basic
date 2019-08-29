# 게이트 순환 유닛(Gated Recurrent Unit, GRU)

GRU는 LSTM의 장기 의존성 문제에 대한 해결책을 유지하면서, 은닉 상태를 업데이트하는 계산을 줄였다. GRU는 성능은 LSTM과 유사하면서 복잡했던 LSTM의 구조를 다시 간단화 시켰다.

LSTM에서는 출력, 입력, 삭제 게이트라는 3개의 게이트가 존재했다. 하지만 GRU에서는 업데이트 게이트와 리셋 게이트 두 가지 게이트만이 존재한다. GRU는 LSTM보다 학습 속도가 빠르다고 알려져있지만 여러 평가에서 GRU는 LSTM과 비슷한 성능을 보인다고 알려져 있다.

```
model.add(GRU(hidden_size, input_shape=(timesteps, input_dim)))
```
