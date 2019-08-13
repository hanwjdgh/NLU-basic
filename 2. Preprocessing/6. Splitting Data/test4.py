import numpy as np
from sklearn.model_selection import train_test_split

X, Y = np.arange(10).reshape((5, 2)), range(5)

print(X)
print(list(Y)) 

"""
→ 
[[0 1]
 [2 3]
 [4 5]
 [6 7]
 [8 9]]
[0, 1, 2, 3, 4]
"""

"""
X : 독립 변수 데이터. (배열이나 데이터프레임)
Y : 종속 변수 데이터. 레이블 데이터.
test_size : 테스트용 데이터 개수를 지정한다. 1보다 작은 실수를 기재할 경우, 비율을 나타낸다.
train_size : 학습용 데이터의 개수를 지정한다. 1보다 작은 실수를 기재할 경우, 비율을 나타낸다.
(test_size와 train_size 중 하나만 기재해도 가능)
random_state : 난수 시드
"""
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=1234)

print(X_train)
print(X_test)

"""
→ 
[[2 3]
 [4 5]
 [6 7]]
[[8 9]
 [0 1]]
"""

print(Y_train)
print(Y_test)

"""
→ 
[1, 2, 3]
[4, 0]
"""