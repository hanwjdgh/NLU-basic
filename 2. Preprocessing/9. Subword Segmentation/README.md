# Subword Segmentation

단어 분리(Subword Segmentation)는 기계가 아직 배운 적이 없는 단어더라도 대처할 수 있도록 도와주는 기법입니다. 이 방법은 이제는 기계 번역 등에서 주요 전처리로 사용되고 있습다.

OOV(Out-Of-Vocabulary) : 테스트 단계에서 기계가 미처 배우지 못한 모르는 단어들

## 1. BPE(Byte Pair Encoding) 알고리즘

연속적으로 가장 많이 등장한 글자의 쌍을 찾아서 하나의 글자로 병합하는 방식의 데이터 압축 알고리즘

ex) test1.py
