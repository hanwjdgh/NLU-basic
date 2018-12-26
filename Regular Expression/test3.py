import re

r=re.compile("ab.")
print(r.search("kkkabc"))
print(r.match("kkkabc"))
print(r.match("abckkk"))