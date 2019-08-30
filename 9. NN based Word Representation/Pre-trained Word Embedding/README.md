# 사전 훈련된 워드 임베딩(Pre-trained Word Embedding)

## 1. 케라스 임베딩 층(Keras Embedding layer)

케라스는 단어들에 대해서 워드 임베딩을 수행하는 도구인 Embedding()을 제공, 임베딩 층을 사용하기 위해서 입력 시퀀스의 각 입력은 모두 정수 인코딩이 되어있어야 한다.

어떤 단어 → 단어에 부여된 고유한 정수값(임베딩 층의 입력) → 임베딩 층 통과 → 밀집 벡터(임베딩 층의 출력)

```
v = Embedding(20001, 256, input_length=500)
```

Embedding()의 인자

- vocab_size : 텍스트 데이터의 전체 단어 집합의 크기
- output_dim : 임베딩이 되고 난 후의 단어의 차원
- input_length : 입력 시퀀스의 길이

Embedding()은 (number of samples, input_length)인 2D 정수 텐서를 입력받고. 이 때 각 sample은 정수 인코딩이 된 결과로, 정수의 시퀀스이다. Embedding()은 워드 임베딩 작업을 수행하고 (number of samples, input_length, embedding word dimentionality)인 3D 실수 텐서를 리턴한다.

ex) test1.py

## 2. 사전 훈련된 글로브 임베딩(Pre-Trained GloVe Embedding) 사용하기

임베딩 벡터를 얻기 위해서 케라스의 Embedding()을 사용하기도 하지만, 때로는 이미 훈련되어져 있는 워드 임베딩을 불러서 이를 임베딩 벡터로 사용하기도 한다.

ex) test2.py
