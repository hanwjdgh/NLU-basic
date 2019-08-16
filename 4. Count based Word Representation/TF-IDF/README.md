# Term Frequency-Inverse Document Frequency

TF-IDF는 단어의 빈도와 역 문서 빈도(문서의 빈도에 특정 식을 취함)를 사용하여 DTM 내의 각 단어들마다 중요한 정도를 가중치로 주는 방법, 사용 방법은 우선 DTM을 만든 후에, 거기에 TF-IDF 가중치를 주면된다.

TF-IDF는 TF와 IDF를 곱한 값을 의미

(1) tf(d,t) : 특정 문서 d에서의 특정 단어 t의 등장 횟수

(2) df(t) : 특정 단어 t가 등장한 문서의 수

(3) idf(d, t) : df(t)에 반비례하는 수 ( idf(d,t) = log(n / 1+df(t)) )

ex) test1.py, test2.py
