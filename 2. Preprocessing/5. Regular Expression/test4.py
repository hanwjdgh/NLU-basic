import re  

text = """100 John    PROF
101 James   STUD
102 Mac   STUD"""  

print(re.split('\s+', text))

"""
→ ['100', 'John', 'PROF', '101', 'James', 'STUD', '102', 'Mac', 'STUD']
  최소 1개의 공백을 찾아 문자열을 자른다.
"""

print(re.findall('\d+',text))

"""
→ ['100', '101', '102]
  숫자를 찾아 문자열로 리턴
"""

print(re.findall('[A-Z]',text))

"""
→ ['J', 'P', 'R', 'O', 'F', 'J', 'S', 'T', 'U', 'D', 'M', 'S', 'T', 'U', 'D']
  대문자의 경우
"""

print(re.findall('[A-Z]{4}',text)) 

"""
→ ['PROF', 'STUD', 'STUD']
  대문자 4개가 존재하는 경우 출력
"""

print(re.findall('[A-Z][a-z]+',text))

"""
→ ['John', 'James', 'Mac'] 
  시작은 대문자 그 후는 소문자 여려개가 나오는 경우
"""

letters_only = re.sub('[^a-zA-Z]', ' ', text)
print(letters_only)

"""
→ 'John    PROF     James   STUD     Mac   STUD'
  영문이 아닌 겨우 ' '로 대체
"""