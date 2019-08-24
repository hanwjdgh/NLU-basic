# Softmax Regression

다중 클래스 분류 문제를 위한 소프트맥스 회귀(Softmax Regression)

## 1. 다중 클래스 분류(Multi-class Classification)

세 개 이상의 답 중 하나를 고르는 문제를 다중 클래스 분류라고 한다.

## 2. 소프트맥스 함수(Softmax function)

소프트맥스 함수는 분류해야하는 정답지(클래스)의 총 개수를 k라고 할 때, k차원의 벡터를 입력받아 각 클래스에 대한 확률을 추정하는 함수이다.

pi = e <sup>z<sub>i</sub></sup> / ∑<sup>k</sup><sub>j=1</sub>e<sup>z<sub>j</sub></sup>  for i=1,2,...k

ex) test1.py
