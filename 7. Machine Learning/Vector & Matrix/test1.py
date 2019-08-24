import numpy as np

# 0차원

d=np.array(5)
print(d.ndim) # 차원수 출력
print(d.shape) # 텐서의 크기 출력 

"""
→ 0
  () #크기가 없음
"""

# 1차원

d=np.array([1, 2, 3, 4])
print(d.ndim)
print(d.shape)

"""
→ 1
  (4,)
"""

# 2차원

d=np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(d.ndim)
print(d.shape)

"""
→ 2
  (3, 4)
"""

# 3차원

d=np.array([
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [10, 11, 12, 13, 14]],
            [[15, 16, 17, 18, 19], [19, 20, 21, 22, 23], [23, 24, 25, 26, 27]]
            ])
print(d.ndim)
print(d.shape)

"""
→ 3
  (2, 3, 5)
"""