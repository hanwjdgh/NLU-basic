# 텍스트 분류(Text Classification)

텍스트 분류(Text Classification)은 텍스트를 입력으로 받아, 텍스트가 어떤 종류의 범주(Class)에 속하는지를 구분하는 작업

텍스트 분류에서 분류해야할 범주가 두 가지라면 이진 분류(Binary Classification)라고 하며, 세 가지 이상이라면 다중 클래스 분류(Multi-Class Classification)라고 한다.

텍스트 분류는 RNN의 다-대-일(Many-to-One) 문제에 속한다.

이진 분류의 문제의 경우 출력층의 활성화 함수로 시그모이드 함수를, 손실 함수로 binary_crossentropy를 사용합니다. 반면, 다중 클래스 문제라면 출력층의 활성화 함수로 소프트맥스 함수를, 손실 함수로 categorical_crossentropy를 사용한다.

또한, 다중 클래스 분류 문제의 경우에는 클래스가 N개라면 출력층에 해당되는 밀집층(dense layer)의 크기는 N이다.

## 1. 스팸 메일 분류하기(Spam Detection)

ex) test1.py

## 2. 로이터 뉴스 분류하기(Reuters News Classification)

ex) test2.py

## 3. IMDB 리뷰 감성 분류하기(IMDB Movie Review Sentiment Analysis)

ex) test3.py

## 4. 나이브 베이즈 분류기(Naive Bayes Classifier)

ex) test4.py

## 5. 네이버 영화 리뷰 감성 분류하기(Naver Movie Review Sentiment Analysis)

ex) test5.py
