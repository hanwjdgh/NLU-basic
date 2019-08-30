# 엘모(Embeddings from Language Model, ELMo)

'언어 모델로 하는 임베딩', ELMo의 가장 큰 특징은 사전 훈련된 언어 모델(Pre-trained language model)을 사용한다는 점

단어를 임베딩하기 전에 전체 문장을 고려해서 임베딩 : 문맥을 반영한 워드 임베딩(Contextualized Word Embedding)

## 1. biLM(Bidirectional Language Model)의 사전 훈련

ELMo는 양쪽 방향의 언어 모델을 둘 다 활용한다고하여 이 언어 모델을 biLM(Bidirectional Language Model)이라고 한다. 기본적으로 다층 구조(Multi-layer)를 전제로 한다.

biLM의 입력이 되는 워드 임베딩 방법으로는 char CNN이라는 방법을 사용, 이 임베딩 방법은 글자(character) 단위로 계산되는데, 이렇게 하면 마치 서브단어(subword)의 정보를 참고하는 것처럼 문맥과 상관없이 dog란 단어와 doggy란 단어의 연관성을 찾아낼 수 있다.

ex) test1.py
