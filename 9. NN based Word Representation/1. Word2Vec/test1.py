import re
from lxml import etree
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models import Word2Vec

targetXML=open('ted_en-20160408.xml', 'r', encoding='UTF8')

target_text = etree.parse(targetXML)
parse_text = '\n'.join(target_text.xpath('//content/text()'))

content_text = re.sub(r'\([^)]*\)', '', parse_text)

sent_text=sent_tokenize(content_text)

normalized_text = []
for string in sent_text:
     tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
     normalized_text.append(tokens)

result=[]
result=[word_tokenize(sentence) for sentence in normalized_text]

#print(result[:10])

"""
[['here', 'are', 'two', 'reasons', 'companies', 'fail', 'they', 'only', 'do', 'more', 'of', 'the', 'same', 'or', 'they', 'only', 'do', 'what', 's', 'new'], 
['to', 'me', 'the', 'real', 'real', 'solution', 'to', 'quality', 'growth', 'is', 'figuring', 'out', 'the', 'balance', 'between', 'two', 'activities', 'exploration', 'and', 'exploitation'], 
['both', 'are', 'necessary', 'but', 'it', 'can', 'be', 'too', 'much', 'of', 'a', 'good', 'thing'], 
['consider', 'facit'], ['i', 'm', 'actually', 'old', 'enough', 'to', 'remember', 'them'], 
['facit', 'was', 'a', 'fantastic', 'company'], ['they', 'were', 'born', 'deep', 'in', 'the', 'swedish', 'forest', 'and', 'they', 'made', 'the', 'best', 'mechanical', 'calculators', 'in', 'the', 'world'], 
['everybody', 'used', 'them'], ['and', 'what', 'did', 'facit', 'do', 'when', 'the', 'electronic', 'calculator', 'came', 'along'], 
['they', 'continued', 'doing', 'exactly', 'the', 'same']]
"""

model = Word2Vec(sentences=result, size=100, window=5, min_count=5, workers=4, sg=0)

a=model.wv.most_similar("man") # 입력한 단어에 대해서 가장 유사한 단어들을 출력
print(a)

"""
[('woman', 0.843902587890625), ('guy', 0.8003332614898682), ('girl', 0.7587348222732544), ('lady', 0.7498583793640137), 
('gentleman', 0.7408038377761841), ('boy', 0.7395448684692383), ('soldier', 0.7041013240814209), ('kid', 0.6777248382568359), 
('friend', 0.6611425876617432), ('poet', 0.6446284055709839)]
"""