import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Sklearn

data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)

print(len(data))
print(data.head(5))

"""
→ 
1103663
   publish_date                                      headline_text
0      20030219  aba decides against community broadcasting lic...
1      20030219     act fire witnesses must be aware of defamation
2      20030219     a g calls for infrastructure protection summit
3      20030219           air nz staff in aust strike for pay rise
4      20030219      air nz strike to affect australian travellers
"""

text = data[['headline_text']]
print(text.head(5))

"""
→
                                       headline_text
0  aba decides against community broadcasting lic...
1     act fire witnesses must be aware of defamation
2     a g calls for infrastructure protection summit
3           air nz staff in aust strike for pay rise
4      air nz strike to affect australian travellers
"""

# 텍스트 전처리

text['headline_text'] = text.apply(lambda row: nltk.word_tokenize(row['headline_text']), axis=1) #
print(text.head(5))

"""
→ 
                                       headline_text
0  [aba, decides, against, community, broadcastin...
1  [act, fire, witnesses, must, be, aware, of, de...
2  [a, g, calls, for, infrastructure, protection,...
3  [air, nz, staff, in, aust, strike, for, pay, r...
4  [air, nz, strike, to, affect, australian, trav...
"""

#nltk.download('stopwords')
stop = stopwords.words('english')
text['headline_text'] = text['headline_text'].apply(lambda x: [word for word in x if word not in (stop)])
tokenized_doc = text['headline_text'].apply(lambda x: [word for word in x if len(word) > 3])
print(text.head(5))
print(tokenized_doc[:5])

"""
→ 
                                       headline_text
0   [aba, decides, community, broadcasting, licence]
1    [act, fire, witnesses, must, aware, defamation]
2     [g, calls, infrastructure, protection, summit]
3          [air, nz, staff, aust, strike, pay, rise]
4  [air, nz, strike, affect, australian, travellers]

0    [decides, community, broadcasting, licence]
1     [fire, witnesses, must, aware, defamation]
2    [calls, infrastructure, protection, summit]
3                    [staff, aust, strike, rise]
4       [strike, affect, australian, travellers
"""

# TF-IDF

# 역토큰화 

detokenized_doc = []
for i in range(len(text)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

text['headline_text'] = detokenized_doc #

vectorizer = TfidfVectorizer(stop_words='english', max_features= 1000) # 상위 1,000개의 단어를 보존 
X = vectorizer.fit_transform(text['headline_text'])
print(X.shape) 

"""
→ (1103663, 1000)
"""

# 토픽 모델링

lda_model=LatentDirichletAllocation(n_components=10,learning_method='online',random_state=777,max_iter=1)
lda_top=lda_model.fit_transform(X)
print(lda_model.components_.shape)

"""
→ (10, 1000)
"""

terms = vectorizer.get_feature_names() # 단어 집합. 1,000개의 단어가 저장됨.

def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])
get_topics(lda_model.components_,terms)

"""
→ 
Topic 1: [('trump', 11977.28), ('government', 9102.6), ('china', 4610.44), ('open', 4158.68), ('island', 3467.29)]
Topic 2: [('australia', 14708.34), ('world', 7414.63), ('canberra', 6613.81), ('house', 4597.48), ('country', 4544.23)]
Topic 3: [('police', 12808.1), ('sydney', 9135.4), ('south', 6499.57), ('coast', 5678.01), ('tasmanian', 4919.86)]
Topic 4: [('election', 8094.72), ('melbourne', 7914.15), ('years', 5829.96), ('family', 4351.41), ('labor', 4014.33)]
Topic 5: [('court', 7060.47), ('charged', 5687.32), ('murder', 5633.21), ('home', 5117.96), ('tasmania', 4668.88)]
Topic 6: [('perth', 6582.05), ('death', 6338.54), ('turnbull', 5065.94), ('life', 4691.18), ('woman', 4278.95)]
Topic 7: [('says', 15181.73), ('queensland', 8486.84), ('north', 6409.77), ('donald', 5979.94), ('brisbane', 5553.89)]
Topic 8: [('adelaide', 7135.37), ('dies', 4972.83), ('indigenous', 4691.12), ('rural', 4571.21), ('hospital', 4413.11)]
Topic 9: [('attack', 5180.65), ('school', 4791.85), ('2016', 4414.85), ('state', 4351.13), ('city', 3465.6)]
Topic 10: [('australian', 11903.25), ('year', 6200.02), ('interview', 5779.75), ('health', 4442.96), ('live', 3904.74)]
"""