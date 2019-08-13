import numpy as np

ar = np.arange(0,16).reshape((4,4))

print(ar)

"""
→ 
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]]
"""

X = ar[ :, :3]
print(X)

"""
→ 
[[ 0  1  2]
 [ 4  5  6]
 [ 8  9 10]
 [12 13 14]]
"""

Y = ar[:,3]
print(Y)

"""
→ [ 3  7 11 15]
"""