import re

# {숫자} 기호 (바로 앞의 문자가 숫자 만큼 존재)

r=re.compile("ab{2}c")
print(r.search("ac")) 
print(r.search("abc")) 
print(r.search("abbc"))
print(r.search("abbbbbc"), end='\n\n')

"""
→ None
  None
  <_sre.SRE_Match object; span=(0, 4), match='abbc'>
  None
"""

# {숫자1, 숫자2} 기호 (바로 앞의 문자가 숫자1 이상 숫자2 이하로 존재)

r=re.compile("ab{2,8}c")
print(r.search("ac"))
print(r.search("ac"))
print(r.search("abbbbbbbbc"), end='\n\n')

"""
→ None
  None
  <_sre.SRE_Match object; span=(0, 10), match='abbbbbbbbc'>
"""

# {숫자, } 기호 (바로 앞의 문자를 숫자만큼 반복)

r=re.compile("a{2,}bc")
print(r.search("bc"))
print(r.search("aa"))
print(r.search("aabc"), end='\n\n')

"""
→ None
  None
  <_sre.SRE_Match object; span=(0, 4), match='aabc'>
"""

# [] 기호 ([]안의 문자들 중 한 개의 문자와 매치)

r=re.compile("[abc]") 
print(r.search("zzz")) 
print(r.search("a"), end='\n\n')

"""
→ None
  <_sre.SRE_Match object; span=(0, 1), match='a'>
"""

r=re.compile("[a-z]")
print(r.search("AAA"))
print(r.search("aBC"), end='\n\n')

"""
→ None
  <_sre.SRE_Match object; span=(0, 1), match='a'>
"""


# [^문자] 기호 (^기호 뒤에 붙은 문자들을 제외한 모든 문자를 매치)

r=re.compile("[^abc]")
print(r.search("a"))
print(r.search("ab"))
print(r.search("b"))
print(r.search("d"))
print(r.search("1"))

"""
→ None
  None
  None
  <_sre.SRE_Match object; span=(0, 1), match='d'>
  <_sre.SRE_Match object; span=(0, 1), match='1'>
"""