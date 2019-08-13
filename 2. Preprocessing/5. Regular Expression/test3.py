import re

# re.search(), re.match()
# search()가 정규 표현식 전체에 대해서 문자열이 매치하는지를 본다면, match()는 문자열의 첫 부분부터 정규 표현식과 매치하는지를 확인합니다. 
# 문자열 중간에 찾을 패턴이 있다고 하더라도, match 함수는 문자열의 시작에서 패턴이 일치하지 않으면 찾지 않습니다.

r=re.compile("ab.")
print(r.search("kkkabc"))
print(r.match("kkkabc"))
print(r.match("abckkk"), end='\n\n')

"""
→ <_sre.SRE_Match object; span=(3, 6), match='abc'>
  None
  <_sre.SRE_Match object; span=(0, 3), match='abc'>
"""

# re.split() (문자열 자르기)

text="사과 딸기 수박 메론 바나나"
print(re.split(" ",text))

text="""사과
딸기
수박
메론
바나나"""
print(re.split("\n",text))

text="사과+딸기+수박+메론+바나나"
print(re.split("\+",text), end='\n\n')

"""
→ ['사과', '딸기', '수박', '메론', '바나나']
  ['사과', '딸기', '수박', '메론', '바나나']  
  ['사과', '딸기', '수박', '메론', '바나나']
"""

# re.findall() (정규 표현식과 매치되는 모든 문자열들을 리스트로 리턴)

text="""이름 : 김철수
전화번호 : 010 - 1234 - 1234
나이 : 30
성별 : 남"""  
print(re.findall("\d+",text))
print(re.findall("\d+", "문자열 입니다."), end='\n\n')

"""
→ ['010', '1234', '1234', '30']
  []
"""

# re.sub() (정규 표현식 패턴과 일치하는 문자열을 찾아 다른 문자열로 대체)

text="Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."
print(re.sub('[^a-zA-Z]',' ',text))

"""
→ 'Regular expression   A regular expression  regex or regexp     sometimes called a rational expression        is  in theoretical computer science and formal language theory  a sequence of characters that define a search pattern '
   알파벳 이외의 문자를 ' '로 대체
"""