# 피드 포워드 신경망 언어 모델(Neural Network Language Model, NNLM)

기존 N-gram 언어 모델이 희소 문제를 가지고 있는데 이를 해결하기 위해서 언어 모델 또한 단어의 유사도를 학습할 수 았도록 설계돤 언어 모델이 신경망 언어 모델이다. 그리고 이 아이디어는 단어 간 유사도를 반영한 벡터를 만드는 워드 임베딩(word embedding)의 아이디어이기도 하다.

NNLM은 단어의 유사도를 단어를 표현하기 위해 밀집 벡터(dense vector)를 사용하므로서 단어의 유사도를 표현할 수 있고 더 이상 모든 n-gram을 저장하지 않아도 된다는 점에서 n-gram 언어 모델보다 저장 공간의 이점을 가진다.

하지만, NNLM은 n-gram 언어 모델과 마찬가지로 다음 단어를 예측하기 위해 모든 이전 단어를 참고하는 것이 아니라, 정해진 n개의 단어만을 참고한다. 이는 버려지는 단어들이 가진 문맥 정보는 참고할 수 없음을 의미
