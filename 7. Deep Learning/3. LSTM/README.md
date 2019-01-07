# LSTM

RNN은 비교적 짧은 시퀀스(sequence)에 대해서만 효과를 보이는 단점이 있습니다. 즉, RNN의 time-step이 길어질 수록 앞의 정보가 뒤로 충분히 전달되지 못하는 현상이 발생 (장기 의존성 문제(the problem of Long-Term Dependencies))

이러한 단점을 보완한 RNN의 일종을 장단기 메모리(Long Short-Term Memory)라고 하며, 줄여서 LSTM