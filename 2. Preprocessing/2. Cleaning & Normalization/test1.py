import re

# 길이가 1~2인 단어들을 정규 표현식을 이용하여 삭제

text = "I was wondering if anyone out there could enlighten me on this car."

shortword = re.compile(r'\W*\b\w{1,2}\b')

print(shortword.sub('', text))

"""
→ was wondering anyone out there could enlighten this car.
"""