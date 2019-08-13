# Integer Encoding

텍스트를 숫자로 바꾸는 작업

인덱스를 부여하는 방법은 여러 가지가 있을 수 있는데 랜덤으로 부여하기도 하지만, 보통은 전처리도 같이 겸하기 위해 단어에 대한 빈도수로 정렬한 뒤에 부여합니다.

## 1. 정수 인코딩

단어에 숫자를 부여하는 방법 중 하나로 단어를 빈도수 순으로 정렬하여 단어 집합(vocabulary)을 만들고, 빈도수가 높은 순서대로 차례로 낮은 숫자부터 정수를 부여하는 방법

ex) test1.py

## 2. 케라스(Keras)의 텍스트 전처리

ex) test2.py

## 3. enumerate

ex) test3.py

## 4. NLTK의 FreqDist 클래스

빈도수 계산 클래스 FreqDist를 지원

ex) test4.py
