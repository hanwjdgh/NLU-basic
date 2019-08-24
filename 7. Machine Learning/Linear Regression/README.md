# Linear Regression

## 1. Linear Regression

x(독립변수), W(가중치), b(편향)

1. 단순 선형 회귀 분석(Simple Linear Regression Analysis)

    y = W<sub>x</sub> + b  ex) test1.py

2. 다중 선형 회귀 분석(Multiple Linear Regression Analysis)

    y = W<sub>1x1</sub> + W<sub>2x2</sub> + ... W<sub>nxn</sub> + b ex) test2.py

## 2. 가설(Hypothesis) 세우기

머신 러닝에서는 y와 x간의 관계를 유추한 식을 가설(Hypothesis)이라고 한다. H(x)= W<sub>x</sub> + b

## 3. 비용 함수(Cost function) : 평균 제곱 오차(MSE)

실제값과 예측값에 대한 오차에 대한 식을 목적 함수(Objective function) 또는 비용 함수(Cost function) 또는 손실 함수(Loss function)라고 한다.

비용 함수는 단순히 실제값과 예측값에 대한 오차를 표현하면 되는 것이 아니라, 예측값의 오차를 줄이는 일에 최적화 된 식이어야 한다. 머신 러닝, 딥 러닝에는 다양한 문제들이 있고, 각 문제들에는 적합한 비용 함수들이 있는데 회귀 문제의 경우에는 주로 평균 제곱 오차(Mean Squered Error, MSE)가 사용된다.

모든 점들과의 오차가 클 수록 평균 제곱 오차는 커지며, 오차가 작아질 수록 평균 제곱 오차는 작아진다. 그러므로 이 평균 최곱 오차. 즉, Cost(W,b)를 최소가 되게 만드는 W와 b를 구하면 결과적으로 y와 x의 관계를 가장 잘 나타내는 직선을 그릴 수 있게 된다.

W,b → minimize cost(W,b)

## 4. 옵티마이저(Optimizer) : 경사하강법(Gradient Descent)

선형 회귀를 포함한 수많은 머신 러닝, 딥 러닝의 학습은 결국 비용 함수를 최소화하는 매개 변수인 W와 b을 찾기 위한 작업을 수행한다. 이때 사용되는 것이 옵티마이저(Optimizer) 알고리즘이다.
