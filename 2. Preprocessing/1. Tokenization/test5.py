import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from konlpy.tag import Kkma  
from konlpy.tag import Okt  

# nltk에서 품사 분류 = pos_tag

text="I am actively looking for Ph.D. students. and you are a Ph.D. student."
x=word_tokenize(text)

print(x)

"""
→ ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D.', 'student', '.']
"""

print(pos_tag(x))

"""
→ [('I', 'PRP'), ('am', 'VBP'), ('actively', 'RB'), ('looking', 'VBG'), ('for', 'IN'), ('Ph.D.', 'NNP'), 
('students', 'NNS'), ('.', '.'), ('and', 'CC'), ('you', 'PRP'), ('are', 'VBP'), ('a', 'DT'), ('Ph.D.', 'NNP'), 
('student', 'NN'), ('.', '.')]
"""

# 한국어 자연어 처리에 사용되는 KoNLPy
"""
1) morphs : 형태소 추출
2) pos : 품사 태깅(Part-of-speech tagging)
3) nouns : 명사 추출
"""

okt=Okt()  
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  
print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  

"""
→ ['열심히', '코딩', '한', '당신', ',', '연휴', '에는', '여행', '을', '가봐요']  
→ [('열심히','Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), 
   ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')] 
→ ['코딩', '당신', '연휴', '여행']  
"""

kkma=Kkma()  
print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))  

"""
→ ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']  
→ [('열심히', 'MAG'), ('코딩', 'NNG'), ('하', 'XSV'), ('ㄴ', 'ETD'), ('당신', 'NP'), (',', 'SP'), ('연휴', 'NNG'), 
   ('에', 'JKM'), ('는', 'JX'), ('여행', 'NNG'), ('을', 'JKO'), ('가보', 'VV'), ('아요', 'EFN')]
→ ['코딩', '당신', '연휴', '여행']  
"""