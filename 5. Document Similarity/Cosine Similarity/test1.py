from numpy import dot
from numpy.linalg import norm
import numpy as np

def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

doc1=np.array([0,1,1,1])
doc2=np.array([1,0,1,1])
doc3=np.array([2,0,2,2])

print(cos_sim(doc1, doc2)) 
print(cos_sim(doc1, doc3)) 
print(cos_sim(doc2, doc3)) 

"""
0.6666666666666667
0.6666666666666667
1.0000000000000002
"""