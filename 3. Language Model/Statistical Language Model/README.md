# Statistical Language Model(SLM)

전통적인 접근 방법인 통계적 언어 모델

## 1. 조건부 확률

조건부 확률은 두 확률 P(A),P(B)에 대해서 아래와 같은 관계를 갖습니다.

p(B|A) = P(A,B) / P(A)

P(A,B) = P(A)P(B|A)

조건부 확률의 연쇄 법칙(chain rule)
P(x1,x2,x3...xn) = P(x1)P(x2|x1)P(x3|x1,x2)...P(xn|x1...xn−1)

ex) 각 단어는 문맥이라는 관계로 인해 이전 단어의 영향을 받아 나온 단어이고 모든 단어로부터 하나의 전체 문장이 완성되기 때문에 조건부확률을 사용한다.

P(An adorable little boy is spreading smiles) = <br/>
P(An) × P(adorable|An) × P(little|An adorable) × P(boy|An adorable little) × P(is|An adorable little boy) × P(spreading|An adorable little boy is) × P(smiles|An adorable little boy is spreading)

## 2. 카운트 기반의 접근

SLM에서 실제 기계는 이전 단어로부터 다음 단어에 대한 확률을 카운트에 기반하여 확률을 계산한다.

ex) An adorable little boy가 나왔을 때, is가 나올 확률인 P(is|An adorable little boy) 계산

P(is|An adorable little boy) = count(An adorable little boy is) / count(An adorable little boy)

한계 : 분모가 0인 경우
해결책 : n-gram 언어 모델과 여러가지 SLM의 일반화기법
