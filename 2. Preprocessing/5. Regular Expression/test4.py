import re  

text = """100 John    PROF
101 James   STUD
102 Mac   STUD"""  

print(re.split('\s+', text))
print(re.findall('\d+',text))
print(re.findall('[A-Z]',text))
print(re.findall('[A-Z]{4}',text)) 
print(re.findall('[A-Z][a-z]+',text))
letters_only = re.sub('[^a-zA-Z]', ' ', text)
print(letters_only)