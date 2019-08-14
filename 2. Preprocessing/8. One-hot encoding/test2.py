from keras_preprocessing.text import Tokenizer

text="나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

t = Tokenizer()

t.fit_on_texts([text]) # [] 형태 주의
print(t.word_index)

"""
→ {'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}
"""

vocab_size = len(t.word_index) 

from keras.utils import to_categorical
x = to_categorical(x, num_classes=vocab_size+1) # 실제 단어 집합의 크기보다 +1로 크기를 만들어야함.
                                                # 자동으로 원-핫 인코딩을 만들어 주는 유용한 도구
print(x)

"""
→
[[0. 0. 1. 0. 0. 0. 0. 0.]  
 [0. 0. 0. 0. 0. 1. 0. 0.]  
 [0. 1. 0. 0. 0. 0. 0. 0.]  
 [0. 0. 0. 0. 0. 0. 1. 0.]  
 [0. 0. 0. 1. 0. 0. 0. 0.]  
 [0. 0. 0. 0. 0. 0. 0. 1.]] 
"""