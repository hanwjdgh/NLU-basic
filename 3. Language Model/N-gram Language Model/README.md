# N-gram Language Model

N-gram 언어 모델은 여전히 카운트에 기반한 통계적 접근을 사용하고 있으므로 SLM의 일종, 이전에 등장한 모든 단어를 고려하는 것이 아니라 일부 단어만 고려하는 접근 방법을 사용

## 1. 코퍼스에서 카운트하지 못하는 경우 감소

SLM의 한계는 훈련 코퍼스에 확률을 계산하고 싶은 문장이나 단어가 없을 수 있다는 점, 확률을 계산하고 싶은 문장이 길어질수록 갖고있는 코퍼스에서 그 문장이 존재하지 않을 가능성이 높다.

마르코프의 가정을 사용

P(is|An adorable little boy) ≈ P(is|boy)

더 짧은 워드 시퀀스(word sequence)가 코퍼스에 있을 가능성이 더 높다.

## 2. N-gram

N-gram은 n개의 연속적인 단어 나열을 의미, 코퍼스에서 N개의 단어 뭉치 단위로 끊어서 이를 하나의 토큰으로 간주

N-gram을 통한 언어 모델에서는 다음에 나올 단어의 예측은 오직 n-1개의 단어에만 의존

ex) An adorable little boy is spreading smiles

unigrams : an, adorable, little, boy, is, spreading, smiles

bigrams : an adorable, adorable little, little boy, boy is, is spreading, spreading smiles

trigrams : an adorable little, adorable little boy, little boy is, boy is spreading, is spreading smiles

4-grams : an adorable little boy, adorable little boy is, little boy is spreading, boy is spreading smiles

## 3. N-gram Language Model의 한계

N-gram은 뒤의 단어 몇 개만 보다 보니 의도하고 싶은 대로 문장을 끝맺음하지 못하는 경우가 생긴다는 점

(1) n을 선택하는 것은 trade-off 문제

n을 너무 크게 선택하면 OOV문제가 발생할 수 있고 n이 커질수록 모델 사이즈는 굉장히 커지게 된다.

n을 너무 작게 선택하면 훈련 코퍼스에서 카운트는 잘 되겠지만 근사의 정확도는 점점 실제의 확률분포와 멀어진다.

trade-off 문제로 인해 정확도를 높이려면 n은 최대 5를 넘게 잡아서는 안 된다고 권장되고 있다.

(2) 카운트 했을 때 0이 되는 문제(zero count problem)

N-gram 언어 모델도, 확률이 0이 되는 또는 확률 자체를 계산할 수 없는 문제를 완전히 피할 수는 없다.

