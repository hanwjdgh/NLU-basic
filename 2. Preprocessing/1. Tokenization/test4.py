import nltk
from nltk.tokenize import sent_tokenize

# 영어 문장의 토큰화 = sent_tokenize

text="His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to mae sure no one was near."
print(sent_tokenize(text))

"""
→ ['His barber kept his word.', 'But keeping such a huge secret to himself was driving him crazy.', 
'Finally, the barber went up a mountain and almost to the edge of a cliff.', 'He dug a hole in the midst of some reeds.', 
'He looked about, to mae sure no one was near.']
"""

text="I am actively looking for Ph.D. students. and you are a Ph.D student."
print(sent_tokenize(text))

"""
→ ['I am actively looking for Ph.D. students.', 'and you are a Ph.D student.']
NLTK는 단순히 온점을 구분자로 하여 문장을 구분하지 않았기 때문에, Ph.D.를 문장 내의 단어로 인식하여 성공적으로 인식한다.
"""