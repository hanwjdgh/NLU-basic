import nltk
import pandas as pd
import gensim
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import stopwords
from gensim import corpora

# Gensim

# 20개의 다른 주제를 가진 뉴스 데이터 = Twenty Newsgroups

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
documents = dataset.data

print(len(documents)) # 뉴스의 수

"""
→ 11314
"""

print(dataset.target_names) # 카테고리 출력

"""
→
['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 
  'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 
  'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 
  'talk.politics.misc', 'talk.religion.misc'] 
"""

# 텍스트 전처리

news_df = pd.DataFrame({'document':documents})

# 특수 문자 제거
news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")

# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))

# 전체 단어에 대한 소문자 변환
news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())

#nltk.download('stopwords')
stop_words = stopwords.words('english') # NLTK로부터 불용어를 받아옵니다.
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split()) # 토큰화
tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])

print(tokenized_doc[:5])

"""
→
0    [well, sure, story, seem, biased, disagree, st...
1    [yeah, expect, people, read, actually, accept,...
2    [although, realize, principle, strongest, poin...
3    [notwithstanding, legitimate, fuss, proposal, ...
4    [well, change, scoring, playoff, pool, unfortu...
"""

# 정수 인코딩과 단어 집합 만들기

dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
print(corpus[1])

"""
→
[(52, 1), (55, 1), (56, 1), (57, 1), (58, 1), (59, 1), (60, 1), (61, 1), (62, 1), (63, 1), (64, 1), (65, 1), (66, 2), 
 (67, 1), (68, 1), (69, 1), (70, 1), (71, 2), (72, 1), (73, 1), (74, 1), (75, 1), (76, 1), (77, 1), (78, 2), (79, 1), 
 (80, 1), (81, 1), (82, 1), (83, 1), (84, 1), (85, 2), (86, 1), (87, 1), (88, 1), (89, 1)]
"""

print(len(dictionary)) # 학습된 단어의 개수

"""
→ 64281
"""

# LDA 모델 훈련시키기

NUM_TOPICS = 20 #20개의 토픽, k=20
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15) # passes 알고리즘 횟수
topics = ldamodel.print_topics(num_words=4) # 총 4개의 단어만 출력
for topic in topics:
    print(topic)

"""
→
(0, '0.023*"would" + 0.013*"think" + 0.013*"like" + 0.012*"people"')
(1, '0.007*"science" + 0.006*"many" + 0.005*"world" + 0.005*"moral"')
(2, '0.013*"game" + 0.013*"team" + 0.011*"year" + 0.009*"games"')
(3, '0.009*"nrhj" + 0.006*"wwiz" + 0.006*"bxom" + 0.006*"gizw"')
(4, '0.015*"contest" + 0.012*"myers" + 0.008*"guidelines" + 0.007*"judges"')
(5, '0.021*"space" + 0.010*"research" + 0.009*"nasa" + 0.009*"university"')
(6, '0.012*"government" + 0.011*"armenian" + 0.010*"president" + 0.008*"turkish"')
(7, '0.008*"filename" + 0.007*"soon" + 0.006*"gordon" + 0.006*"pitt"')
(8, '0.011*"jesus" + 0.008*"people" + 0.008*"would" + 0.008*"believe"')
(9, '0.030*"drive" + 0.020*"disk" + 0.018*"scsi" + 0.013*"hard"')
(10, '0.012*"said" + 0.008*"people" + 0.008*"know" + 0.007*"went"')
(11, '0.017*"thanks" + 0.017*"windows" + 0.014*"anyone" + 0.014*"know"')
(12, '0.013*"information" + 0.010*"mail" + 0.009*"data" + 0.008*"list"')
(13, '0.009*"israel" + 0.008*"state" + 0.007*"government" + 0.007*"jews"')
(14, '0.006*"used" + 0.006*"also" + 0.005*"good" + 0.005*"like"')
(15, '0.024*"file" + 0.015*"window" + 0.015*"program" + 0.014*"output"')
(16, '0.016*"printf" + 0.016*"char" + 0.006*"string" + 0.006*"compound"')
(17, '0.016*"image" + 0.014*"available" + 0.012*"version" + 0.011*"graphics"')
(18, '0.016*"water" + 0.012*"henrik" + 0.007*"plastic" + 0.004*"homeland"')
(19, '0.010*"health" + 0.008*"medical" + 0.006*"disease" + 0.005*"pain"')
"""

# LDA 시각화

"""
pip install pyLDAvis

import pyLDAvis.gensim

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.display(vis)
"""

# 문서 별 토픽 분포 보기

def make_topictable_per_doc(ldamodel, corpus, texts):
    topic_table = pd.DataFrame()

    # 몇 번째 문서인지를 의미하는 문서 번호와 해당 문서의 토픽 비중을 한 줄씩 꺼내온다.
    for i, topic_list in enumerate(ldamodel[corpus]):
        doc = topic_list[0] if ldamodel.per_word_topics else topic_list            
        doc = sorted(doc, key=lambda x: (x[1]), reverse=True)
        # 각 문서에 대해서 비중이 높은 토픽순으로 토픽을 정렬한다.
        # EX) 정렬 전 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (10번 토픽, 5%), (12번 토픽, 21.5%), 
        # Ex) 정렬 후 0번 문서 : (2번 토픽, 48.5%), (8번 토픽, 25%), (12번 토픽, 21.5%), (10번 토픽, 5%)
        # 48 > 25 > 21 > 5 순으로 정렬이 된 것.

        # 모든 문서에 대해서 각각 아래를 수행
        for j, (topic_num, prop_topic) in enumerate(doc): #  몇 번 토픽인지와 비중을 나눠서 저장한다.
            if j == 0:  # 정렬을 한 상태이므로 가장 앞에 있는 것이 가장 비중이 높은 토픽
                topic_table = topic_table.append(pd.Series([int(topic_num), round(prop_topic,4), topic_list]), ignore_index=True)
                # 가장 비중이 높은 토픽과, 가장 비중이 높은 토픽의 비중과, 전체 토픽의 비중을 저장한다.
            else:
                break
    return(topic_table)

topictable = make_topictable_per_doc(ldamodel, corpus, tokenized_doc)
topictable = topictable.reset_index() # 문서 번호을 의미하는 열(column)로 사용하기 위해서 인덱스 열을 하나 더 만든다.
topictable.columns = ['문서 번호', '가장 비중이 높은 토픽', '가장 높은 토픽의 비중', '각 토픽의 비중']