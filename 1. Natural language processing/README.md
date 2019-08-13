# Natural language processing

- Natural language : 우리가 일상 생활 속에서 사용하는 언어
- Natural language processing : 자연어를 의미를 분석하여 컴퓨터가 처리할 수 있도록 하는 일

## Development Environment

1. Packages
   1. Anaconda : 파이썬 배포판
   2. nltk, nltk data
   3. NoNLpy : 형태소 분석기
   4. JPype : Java와 Python 연결

2. Framework and Library
   - Tensorflow, Keras, Numpy, Scikit-learn, Gensim

3. Analysis tools
   - Pandas, Matplotlib

## Machine Learing Workflow

1. 수집(Acquisition)

    머신 러닝을 하기위해 기계에 학습시켜야할 데이터(=코퍼스(corpus)))를 수집

2. 점검 및 탐색(Inspection and exploration)

    수집된 데이터의 구조, 노이즈 데이터, 머신 러닝 적용을 위해서 어떻게 정제해햐하는지 들을 파악

    이 단계를 탐색적 데이터 분석(Exploratory Data Analysis, EDA) 단계라고도 하는데 이는 독립 변수, 종속 변수, 변수 유형, 변수의 데이터 타입 등을 점검하며 데이터의 특징과 내재하는 구조적 관계를 알아내는 과정을 의미합니다. 이 과정에서 시각화와 간단한 통계 테스트를 진행하기도 합니다.

3. 전처리 및 정제(Preprocessing and Cleaning)

4. 모델링 및 훈련(Modeling and Training)

    머신 러닝에 대한 코드를 작성하는 단계
    전처리가 완료 된 데이터를 머신 러닝 알고리즘을 통해 기계에게 학습시킨다. 여기서 주의해야할 점은 대부분의 경우에서 모든 데이터를 기계에게 학습시켜서는 안 된다는 점

5. 평가(Evaluation)

    기계가 다 학습이 되었다면 테스트용 데이터로 성능을 평가하게 됩니다. 평가 방법은 기계가 예측한 데이터가 테스트용 데이터의 실제 정답과 얼마나 가까운지를 측정

6. 배포(Deployment)

    평가 단계에서 기계가 성공적으로 훈련이 된 것으로 판단된다면, 완성된 모델이 배포되는 단계
