import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv('movies_metadata.csv', low_memory=False) # 데이터 불러오기
data=data.head(2000) # 전체 데이터중 2000개만 사용 

tfidf = TfidfVectorizer(stop_words='english') # 불용어 제거후 TF-IDF

# print(data['overview'].isnull().sum()) overview 열에 Null 값의 합
data['overview'] = data['overview'].fillna('') # overview에서 Null 값을 가진 경우에는 값 제거

tfidf_matrix = tfidf.fit_transform(data['overview']) # overview에 대해서 tf-idf 수행
print(tfidf_matrix.shape)

"""
→ (2000, 13918)
"""

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix) # 코사인 유사도
indices = pd.Series(data.index, index=data['title']).drop_duplicates() # 영화의 타이틀과 인덱스를 가진 테이블
print(indices.head())

"""
→
title
Toy Story                      0
Jumanji                        1
Grumpier Old Men               2
Waiting to Exhale              3
Father of the Bride Part II    4
dtype: int64
"""

def get_recommendations(title, cosine_sim=cosine_sim):
    # 선택한 영화의 타이틀로부터 해당되는 인덱스를 받아옵니다. 이제 선택한 영화를 가지고 연산할 수 있습니다
    idx = indices[title]

    # 모든 영화에 대해서 해당 영화와의 유사도를 구합니다
    sim_scores = list(enumerate(cosine_sim[idx]))

    # 유사도에 따라 영화들을 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 가장 유사한 10개의 영화
    sim_scores = sim_scores[1:11]

    # 가장 유사한 10개의 영화의 인덱스
    movie_indices = [i[0] for i in sim_scores]

    # 가장 유사한 10개의 영화의 제목
    return data['title'].iloc[movie_indices]

print(get_recommendations('Batman'))

"""
→
1491                      Batman & Robin
1328                      Batman Returns
150                       Batman Forever
1912                                Dune
1601               Chairman of the Board
1678                           B. Monkey
1869    Friday the 13th: A New Beginning
39              Cry, the Beloved Country
1143                        Mediterraneo
956                 The Pompatus of Love
Name: title, dtype: object
"""