# Machine Learning

딥 러닝을 포함하고 있다.

기존의 프로그래밍들은 한계를 가지고 있다. 예를 들면 사진을 분류하는 경우 공통된 명확한 특징을 잡는 것이 쉽지 않다.

머신 러닝은 주어진 데이터로부터 결과를 찾는 것에 초점을 맞추는 것이 아니라, 주어진 데이터로부터 규칙성을 찾는 것에 초점이 맞추어져 있다. 주어진 데이터로부터 규칙성을 찾는 과정 = training

일단 규칙성을 발견해내면, 그 후에 들어오는 새로운 데이터에 대해서 기존의 프로그래밍보다 더 정확한 결과를 예측

## 1. 머신 러닝 모델의 평가

모델을 평가하기 위해서는 데이터를 훈련용, 검증용, 테스트용 이렇게 세 가지로 분리하는 것이 일반적

검증용 데이터는 모델의 성능을 조정하기 위한 용도

하이퍼파라미터란 값에 따라서 모델의 성능에 영향을 주는 매개변수, 사용자가 직접 정해줄 수있는 변수

매개변수는 가중치와 편향과 같은 학습을 통해 바뀌어져가는 변수, 모델의 학습과정에서 얻어지는 값

## 2. 분류(Classification)와 회귀(Regression)

1. 이진 분류 문제(Binary Classification)

    이진 분류는 주어진 입력에 대해서 둘 중 하나의 답을 정하는 문제

2. 다중 클래스 분류(Multi-class Classification)

    다중 클래스 분류는 주어진 입력에 대해서 두 개 이상의 정해진 선택지 중에서 답을 정하는 문제

3. 회귀 문제(Regression)

    회귀 문제는 분류 문제처럼 0 또는 1이나 과학 책장, IT 책장 등과 같이 분리된(비연속적인) 답이 결과가 아니라 연속된 값을 결과로 가진다.

## 3. 지도 학습(Supervised Learning)과 비지도 학습(Unsupervised Learning)

1. 지도 학습

    지도 학습이란 레이블(Label)이라는 정답과 함께 학습하는 것

    ex) 단어 분리

2. 비지도 학습

    비지도 학습은 레이블이 없이 학습하는 것

    ex) LDA, Word2Vec

## 4. 샘플(Sample)과 특성(Feature)

많은 머신 러닝 문제가 1개 이상의 독립 변수 x를 가지고 종속 변수 y를 예측하는 문제입니다. 많은 머신 러닝 모델들, 특히 인공 신경망 모델은 독립 변수, 종속 변수, 가중치, 편향 등을 행렬 연산을 통해 연산하는 경우가 많다.

신 러닝에서는 하나의 데이터, 하나의 행을 샘플(Sample)이라고 부른다. (데이터베이스에서는 레코드라고 부르는 단위) 종속 변수 y를 예측하기 위한 각각의 독립 변수 x를 특성(Feature)이라고 부른다.

## 5. 혼동 행렬(Confusion Matrix)

머신 러닝에서는 맞춘 문제수를 전체 문제수로 나눈 값을 정확도(Accuracy)라고 한다. 하지만 정확도는 맞춘 결과와 틀린 결과에 대한 세부적인 내용을 알려주지는 않기 때문에 이를 위해서 사용하는 것이 혼동 행렬(Confusion Matrix)이다.

예를 들어 양성(Positive)과 음성(Negative)을 구분하는 이진 분류가 있다고 하였을 때 혼동 행렬은 다음과 같고 각 열은 예측값을 나타내며, 각 행은 실제값을 나타낸다.

|-| 참 | 거짓 |
|:--------|:--------:|:--------:|
| 참 | TP | FN |
| 거짓 |FP | TN |

이를 각각 TP(True Positive), TN(True Negative), FP(False Postivie), FN(False Negative)라고 하는데 True는 정답을 맞춘 경우고 False는 정답을 맞추지 못한 경우이다. 그리고 Positive와 Negative는 각각 제시했던 정답입니다.

TP = 양성(Postive)이라고 대답하였고 실제로 양성이라서 정답

TN = 음성(Negative)이라고 대답하였는데 실제로 음성이라서 정답

FP = 양성이라고 대답하였는데, 음성이라서 정답을 틀림

FN = 음성이라고 대답하였는데 양성이라서 정답을 틀림

정밀도(Precision) = 양성이라고 대답한 전체 케이스에 대한 TP의 비율, TP / ( TP + FP )

재현률(Recall) = 실제값이 양성인 데이터의 전체 개수에 대해서 TP의 비율, TP / ( TP + FN )

## 6. 과적합(Overfitting)과 과소 적합(Underfitting)

과적합(Overfitting)이란 훈련 데이터를 과하게 학습한 경우, 과하게 학습하면 테스트 데이터나 실제 서비스에서의 데이터에 대해서는 정확도가 좋지 않은 현상이 발생할 수 있다.

테스트 데이터의 성능이 올라갈 여지가 있음에도 훈련을 덜 한 상태를 반대로 과소적합(Underfitting)이라 한다.
