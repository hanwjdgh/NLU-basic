import re

# .기호
r = re.compile("a.c")
print(r.search("kkk"))
print(r.search("abc"), end='\n\n')

# ?기호
r=re.compile("ab?c")
print(r.search("abbc")) 
print(r.search("abc"))
print(r.search("ac"), end='\n\n')

# *기호
r=re.compile("ab*c")
print(r.search("a"))
print(r.search("ac"))
print(r.search("abc")) 
print(r.search("abbbbc"), end='\n\n')

# +기호
r=re.compile("ab+c")
print(r.search("ac"))
print(r.search("abc"))
print(r.search("abbbbc"), end='\n\n') 

# ^기호
r=re.compile("^a")
print(r.search("bbc"))
print(r.search("ab"), end='\n\n')  