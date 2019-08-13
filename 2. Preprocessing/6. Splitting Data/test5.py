import numpy as np
X, Y = np.arange(0,24).reshape((12,2)), range(12)

print(X)
print(list(Y))

"""
→ 
[[ 0  1]
 [ 2  3]
 [ 4  5]
 [ 6  7]
 [ 8  9]
 [10 11]
 [12 13]
 [14 15]
 [16 17]
 [18 19]
 [20 21]
 [22 23]]
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
"""

n_of_train = int(len(X) * 0.8) # 데이터의 전체 길이의 80%에 해당하는 길이값을 구한다.
n_of_test = int(len(X) - n_of_train) # 전체 길이에서 80%에 해당하는 길이를 뺀다.
print(n_of_train)
print(n_of_test)

"""
→ 9
  3
"""

X_test = X[n_of_train:] #전체 데이터 중에서 20%만큼 뒤의 데이터 저장
Y_test = Y[n_of_train:] #전체 데이터 중에서 20%만큼 뒤의 데이터 저장
X_train = X[:n_of_train] #전체 데이터 중에서 80%만큼 앞의 데이터 저장
Y_train = Y[:n_of_train] #전체 데이터 중에서 80%만큼 앞의 데이터 저장

print(X_test)
print(list(Y_test))

"""
→ 
[[18 19]
 [20 21]
 [22 23]]
[9, 10, 11]
"""