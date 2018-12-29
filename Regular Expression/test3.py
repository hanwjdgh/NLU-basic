import re

# re.match(), re.search()
r=re.compile("ab.")
print(r.search("kkkabc"))
print(r.match("kkkabc"))
print(r.match("abckkk"), end='\n\n')

# re.split()
text="사과 딸기 수박 메론 바나나"
print(re.split(" ",text), end='\n\n')

# re.findall()
text="""이름 : 김철수
전화번호 : 010 - 1234 - 1234
나이 : 30
성별 : 남"""  
print(re.findall("\d+",text), end='\n\n')

# re.sub()
text="Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."
print(re.sub('[^a-zA-Z]',' ',text))