import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from konlpy.tag import Kkma  

# nltk에서 품사 분류 = pos_tag

text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
x=word_tokenize(text)
print(pos_tag(x))

"""
-> [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), 
('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), 
('student', 'NN'), ('.', '.')]
"""

kkma=Kkma()  
print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

"""
-> [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), 
('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
"""