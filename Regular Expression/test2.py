import re

# {숫자} 기호
r=re.compile("ab{2}c")
print(r.search("ac")) 
print(r.search("abc")) 
print(r.search("abbc"))
print(r.search("abbbbbc"), end='\n\n')

# {숫자1, 숫자2} 기호
r=re.compile("ab{2,8}c")
print(r.search("ac"))
print(r.search("ac"))
print(r.search("abbbbbbbbc"), end='\n\n')

# {숫자, } 기호
r=re.compile("a{2,}bc")
print(r.search("bc"))
print(r.search("aa"))
print(r.search("aabc"), end='\n\n')

# [] 기호
r=re.compile("[abc]") 
print(r.search("zzz")) 
print(r.search("a"), end='\n\n')

# [^문자] 기호
r=re.compile("[a-z]")
print(r.search("AAA"))
print(r.search("aBC"), end='\n\n')