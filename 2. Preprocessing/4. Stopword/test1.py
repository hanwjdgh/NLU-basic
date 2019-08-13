import nltk  
from nltk.corpus import stopwords  

# 불용어 확인하기

# NLTK에서는 100여개 이상의 영어 단어들을 불용어로 패키지 내에서 미리 정의
# nltk.download('stopwords')

print(stopwords.words('english')[:10])

"""
→ ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're"]
"""