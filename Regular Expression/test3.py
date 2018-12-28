import re

r=re.compile("ab.")
print(r.search("kkkabc"))
print(r.match("kkkabc"))
print(r.match("abckkk"))

text="사과 딸기 수박 메론 바나나"
print(re.split(" ",text))

