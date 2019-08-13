X,y = zip(['a', 1], ['b', 2], ['c', 3])
print(X)
print(y)

"""
→ ('a', 'b', 'c')
  (1, 2, 3)
"""

# 리스트의 리스트 또는 행렬 또는 뒤에서 배울 개념인 2D 텐서.

sequences=[['a', 1], ['b', 2], ['c', 3]] 
X,y = zip(*sequences) # *를 추가 (반복)
print(X)
print(y)

"""
→ ('a', 'b', 'c')
  (1, 2, 3)
"""