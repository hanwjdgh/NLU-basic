from nltk import FreqDist

test_list = ['barber', 'barber', 'person', 'barber', 'good', 'person']

# FreqDist 클래스는 단어를 키(key), 출현빈도를 값(value)으로 가지는 파이썬의 딕셔너리(dict) 자료형의 형태

fdist = FreqDist(test_list)

print(fdist.N()) # 전체 단어의 수

"""
→ 6
"""

print(fdist.freq("barbar")) # 단어의 확률

"""
→ 0.5
"""

print(fdist["barbar"]) # 단어의 빈도수

"""
→ 3
"""

print(fdist.most_common(2)) # 빈도수가 높은 상위 2개

"""
→ [('barber', 3), ('person', 2)]
"""