import numpy as np

# 벡터와 행렬의 덧셈과 뺄셈

a = np.array([8, 4, 5])
b = np.array([1, 2, 3])
print(a+b)
print(a-b)

"""
→ [9 6 8]
  [7 2 2]
"""

a = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
b = np.array([[5, 6, 7, 8],[1, 2, 3, 4]])
print(a+b)
print(a-b)

"""
→ 
[[15 26 37 48]
 [51 62 73 84]]
[[ 5 14 23 32]
 [49 58 67 76]]
"""

# 벡터의 내적과 행렬의 곱셈

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(np.dot(a,b))

"""
→ 32
"""

a = np.array([[1, 3],[2, 4]])
b = np.array([[5, 7],[6, 8]])
print(np.matmul(a,b))

"""
→ [[23 31]
   [34 46]]
"""