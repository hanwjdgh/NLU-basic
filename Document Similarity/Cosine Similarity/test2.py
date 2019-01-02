import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv('movies_metadata.csv', low_memory=False)
data=data.head(20000)

tfidf = TfidfVectorizer(stop_words='english')
data['overview'] = data['overview'].fillna('')

tfidf_matrix = tfidf.fit_transform(data['overview'])
print(tfidf_matrix.shape)