from numpy import dot
from numpy.linalg import norm
import numpy as np


"""
문서1 : 저는 사과 좋아요
문서2 : 저는 바나나 좋아요
문서3 : 저는 바나나 좋아요 저는 바나나 좋아요

-	바나나	사과	저는	좋아요
문서1	0	1	1	1
문서2	1	0	1	1
문서3	2	0	2	2
"""

def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

doc1=np.array([0,1,1,1])
doc2=np.array([1,0,1,1])
doc3=np.array([2,0,2,2])

print(cos_sim(doc1, doc2)) # 문서1과 문서2의 유사도
print(cos_sim(doc1, doc3)) # 문서1과 문서3의 유사도
print(cos_sim(doc2, doc3)) # 문서2와 문서3의 유사도 

"""
→
0.6666666666666667
0.6666666666666667
1.0000000000000002
"""