# Tagging Task

자연어 처리 분야에서 각 단어가 어떤 유형에 속해있는지를 알아내는 작업, 각 단어의 유형이 사람, 장소, 단체 등 어떤 유형인지를 알아내는 개체명 인식(Named Entity Recognition)과 각 단어의 품사가 명사, 동사, 형용사 인지를 알아내는 품사 태깅(Part-of-Speech Tagging)이 있다.

## 1. 케라스를 이용한 태깅 작업 개요(Tagging Task using Keras)

개체명 인식기와 품사 태거의 공통점은 RNN의 다-대-다(Many-to-Many) 작업이면서 또한 앞, 뒤 시점의 입력을 모두 참고하는 양방향 RNN(Bidirectional RNN)을 사용한다는 점이다.

태깅 작업은 지도 학습에 속한다. X와 y데이터의 쌍(pair)은 병렬 구조를 가진다는 특징과 데이터 길이가 같다.

## 2. 개체명 인식(Named Entity Recognition)

어떤 이름을 의미하는 단어를 보고는 그 단어가 어떤 유형인지를 인식하는 것

ex) test1.py

## 3. 양방향 LSTM을 이용한 개체명 인식(Named Entity Recognition using Bi-LSTM)

### 1. BIO 표현

B는 Begin의 약자로 개체명이 시작되는 부분, I는 Inside의 약자로 개체명의 내부 부분을 의미하며, O는 Outside의 약자로 개체명이 아닌 부분을 의미

ex) test2.py

## 4. 양방향 LSTM을 이용한 품사 태깅(Part-of-speech Tagging using Bi-LSTM)

ex) test3.py

## 5. 양방향 LSTM과 CRF(Bidirectional LSTM + CRF)

