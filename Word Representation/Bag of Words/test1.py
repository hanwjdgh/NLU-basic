from konlpy.tag import Twitter
import re

okt = Twitter()

token = re.sub("(\.)", "", "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")
token = okt.morphs(token)

word2index = {}
bow = []
for voca in token:
        if voca not in word2index.keys():
            word2index[voca] = len(word2index)
            bow.insert(len(word2index), 1)
        else:
            index=word2index.get(voca)
            bow[index]=bow[index]+1

print(word2index)  

"""
-> {'정부': 0, '가': 1, '발표하는': 2, '물가상승률': 3, '과': 4, '소비자': 5, '느끼는': 6, '은': 7, '다르': 8, '다': 9}
"""

print(bow)

"""
-> [1, 2, 1, 2, 1, 1, 1, 1, 1, 1]
"""