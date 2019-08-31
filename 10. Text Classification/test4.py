from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

newsdata=fetch_20newsgroups(subset='train')
print(newsdata.keys())
print (len(newsdata.data), len(newsdata.filenames), len(newsdata.target_names), len(newsdata.target))

""" 
→
dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR', 'description'])
11314 11314 20 11314
"""

dtmvector = CountVectorizer()
X_train_dtm = dtmvector.fit_transform(newsdata.data)
print(X_train_dtm.shape)

""" 
→ (11314, 130107)
"""

tfidf_transformer = TfidfTransformer()
tfidfv = tfidf_transformer.fit_transform(X_train_dtm)
print(X_train_dtm.shape)

""" 
→ (11314, 130107)
"""

mod = MultinomialNB()
mod.fit(tfidfv, newsdata.target)
MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

newsdata_test = fetch_20newsgroups(subset='test', shuffle=True) #테스트 데이터 갖고오기
X_test_dtm = dtmvector.transform(newsdata_test.data) #테스트 데이터를 DTM으로 변환
tfidfv_test = tfidf_transformer.transform(X_test_dtm) #DTM을 TF-IDF 행렬로 변환

predicted = mod.predict(tfidfv_test) #테스트 데이터에 대한 예측
print("정확도:", accuracy_score(newsdata_test.target, predicted)) #예측값과 실제값 비교

""" 
→ 정확도: 0.7738980350504514
"""