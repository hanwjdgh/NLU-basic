import nltk  
from nltk.corpus import stopwords  

# 불용어 확인하기

#nltk.download('stopwords')
print(stopwords.words('english')[:10])

"""
-> ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]
"""