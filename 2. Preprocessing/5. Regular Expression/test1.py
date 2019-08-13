import re

# .기호 (1개의 문자)

r = re.compile("a.c")
print(r.search("kkk"))
print(r.search("ac"))
print(r.search("abc"), end='\n\n')

"""
→ None
  None
  <_sre.SRE_Match object; span=(0, 3), match='abc'>
"""

# ?기호 (바로 앞의 문자가 0개 또는 1개의 문자)

r=re.compile("ab?c")
print(r.search("abbc")) 
print(r.search("abc"))
print(r.search("ac"), end='\n\n')

"""
→ None
  <_sre.SRE_Match object; span=(0, 3), match='abc'>
  <_sre.SRE_Match object; span=(0, 2), match='ac'>
"""

# *기호 (바로 앞의 문자가 0개 이상의 문자)

r=re.compile("ab*c")
print(r.search("a"))
print(r.search("ac"))
print(r.search("abc")) 
print(r.search("abbbbc"), end='\n\n')

"""
→ None
  <_sre.SRE_Match object; span=(0, 2), match='ac'>
  <_sre.SRE_Match object; span=(0, 3), match='abc'>
  <_sre.SRE_Match object; span=(0, 6), match='abbbbc'>
"""

# +기호 (바로 앞의 문자가 최소 1개 이상)

r=re.compile("ab+c")
print(r.search("ac"))
print(r.search("abc"))
print(r.search("abbbbc"), end='\n\n') 

"""
→ None
  <_sre.SRE_Match object; span=(0, 3), match='abc'>
  <_sre.SRE_Match object; span=(0, 6), match='abbbbc'>  
"""

# ^기호 (바로 뒤의 문자로 시작되는 문자열)

r=re.compile("^a")
print(r.search("bbc"))
print(r.search("ab"), end='\n\n')  

"""
→ None
  <_sre.SRE_Match object; span=(0, 1), match='a'>
"""