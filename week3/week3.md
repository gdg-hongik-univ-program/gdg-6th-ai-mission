Week1~2에서는 우리가 직접 모델을 설계하고, 직접 학습까지 해보았습니다.

그런데 여기서 이런 의문이 들 수 있습니다.

> 굳이 처음부터 모델을 만들어야 할까?
> 

> 
> 
> 
> 이미 엄청난 양의 데이터를 학습해서, **언어를 잘 이해하는 ChatGPT나 Gemini 같은 똑똑한 모델**이 있는데,그 모델을 **그대로 가져와서**, 지금 우리가 풀고 싶은 문제에 맞게 **조금만 다시 가르치면 되지 않을까?**
> 

Week 3에서는 이렇게 이미 학습된 모델을 가져와, 우리 데이터로 다시 학습시키는 방법을 파인튜닝(Fine-Tuning) 을 학습해볼 예정입니다.

그리고 또 이런 문제도 생길 수도 있습니다. 

> “모델이 학습하지 않은 최신 정보는 어떻게 알게 할까?”
> 

이 문제를 해결하기 위해 등장한 방식이 RAG입니다.

이번 주차에서는 다음 두 가지를 학습해보아요!

1. Fine-Tuning
2. RAG

---

# 이번 주차의 학습 목표

- 사전학습 모델(Pretrained Model)의 개념 이해
- Transformer 기반 언어 모델의 동작 방식 이해
- BERT 기반 Fine-Tuning 구조 이해
- RAG(Retrieval-Augmented Generation)의 전체 흐름 이해
- Fine-Tuning과 RAG의 차이 이해

# Mission1 (fine-tuning)

우리가 영화 리뷰를 긍정/부정으로 분류하는 AI를 만들고 싶다고 가정해봅시다.

그런데 직접 모델을 처음부터 학습시키려면:

- 엄청 많은 데이터
- 긴 학습 시간
- 큰 GPU 자원

이 필요합니다.

하지만 이미 한국어를 이해하는 BERT 모델이 존재합니다.

그러면 우리는:

> “한국어는 이미 알고 있으니까, 영화 리뷰만 조금 더 학습시키자!”
> 

라는 전략을 사용할 수 있습니다.

이것이 Fine-Tuning입니다.

## 사전학습 모델 불러오기

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

MODEL_NAME = "klue/bert-base"
```

한국어 BERT 모델을 불러옵니다.

- [ ]  왜 우리는 모델을 처음부터 만들지 않고, 이미 학습된 모델을 가져다 사용할까요?
- [ ]  Transformer 모델이 무엇인지 정리해주세요.
- [ ]  BERT 모델 구조를 간단하게 정리해주세요.

## Tokenizer 이해하기

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
```

week1에서 모델은 문장을 직접 읽지 못하여 숫자로 바꿔줘야한다는 것을 배웠었는데요. 이 역할을 하는 것이 Tokenizer입니다.

아래 문장을 tokenizer에 넣고 결과를 확인해보세요.

```python
sample_text = "이 영화 정말 재미있다"

encoded = tokenizer(sample_text)

print(encoded)
```

- [ ]  출력 결과에서 input_ids는 무엇일까요?

## Fine-Tuning 모델 불러오기

```python
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)
```

이번에는 감정 분석을 위한 모델을 가져오도록 하겠습니다. 
기존 BERT는 “한국어를 이해하는 AI”이지만, 우리는 “긍정/부정을 분류하는 AI” 가 필요합니다. 그래서 마지막에 “긍정/부정 판단기”를 붙여주어야 합니다. 

- [ ]  왜 `num_labels=2`일까요?

## 모델 구조 확인하기

```python
print(model)
```

모델 구조를 출력해보면 매우 복잡한 출력 결과를 확인해볼 수 있을거에요. 

먼저 BERT가 문장을 어떻게 처리하는지 이해해봅시다.

BERT는 문장을 이렇게 처리합니다. 

```markdown
“이 영화 재미있다”
↓
문장을 숫자로 바꿈
↓
단어들의 관계를 살펴봄
↓
문장 전체 의미를 정리함
↓
긍정인지 부정인지 판단함
```

이 흐름을 실제 모델 구조와 연결해보면,

```markdown
1. 문장을 숫자로 바꿈
→ embeddings: BertEmbeddings

2. 단어들의 관계를 살펴봄
→ encoder: BertEncoder
→ attention: BertAttention
→ self: BertSelfAttention

3. 문장 전체 의미를 정리함
→ pooler: BertPooler

4. 긍정인지 부정인지 판단함
→ classifier: Linear(in_features=768, out_features=2)
```

---

### 1. Embedding: 문장을 숫자로 바꾸는 단계

모델 구조 출력 결과에서 아래 부분을 찾아보세요.

```
(embeddings): BertEmbeddings
```

Transformer 모델은 문장을 글자 그대로 처리하지 못합니다.

그래서 먼저 단어를 숫자 벡터로 변환합니다.

예를 들어 `"오늘 날씨 좋다"`라는 문장은 모델 안에서 `[0.12, -0.44, 0.91, ...]` 와 같은 숫자 형태로 바뀝니다. 이 과정을 Embedding이라고 합니다.

출력 코드에서 아래 부분들을 찾아보세요.

```
word_embeddings -> 단어 뜻을 숫자로 바꿈
position_embeddings -> 단어 위치를 숫자로 바꿈
token_type_embeddings -> 문장 구분 정보를 숫자로 바꿈
```

화살표 옆의 간단한 설명을 참고하여, 각각의 embedding layer가 어떠한 역할을 하는지 정리해주세요. 

- [ ]  `word_embeddings`, `position_embeddings`, `token_type_embeddings` 의 역할에 대해서 정리해주세요.

---

### 2. Attention: 단어 간 관계를 이해하는 단계

모델 구조 출력 결과에서 아래 부분을 찾아보세요.

```
(attention): BertAttention
(self): BertSelfAttention
```

Attention은 문장 속 단어들이 서로 얼마나 관련 있는지 계산하는 구조입니다. 

예를 들어 `"오늘 날씨가 좋아서 기분이 좋다"`라는 문장에서 `날씨 ↔ 좋다` , `기분 ↔ 좋다`같은 관계를 함께 봅니다. 즉, 단어를 따로따로 이해하지 않고, 단어 간 관계, 문맥, 문장 전체 흐름을 함께 고려합니다.

이렇게 문맥을 이해하는 것이 매우 중요한 또 다른 이유가 있습니다. 예를 들어, `배를 먹는다`  `배를 탄다`두 문장 모두 `배`라는 단어가 등장하지만 의미는 다릅니다. 첫 번째 문장에서 `배`는 과일을 의미하고, 두 번째 문장에서 `배`는 탈것을 의미합니다. BERT는 `배`라는 단어만 따로 보는 것이 아니라, 주변에 있는 `먹는다`, `탄다` 같은 단어들과의 관계를 함께 보며 현재 문맥에서 어떤 의미로 사용되었는지 이해합니다.

이처럼 Transformer 모델은 단어 자체만 보는 것이 아니라, 단어 간 관계와 문맥을 함께 고려하여 문장의 의미를 이해합니다.

- [ ]  Attention 구조에 대해서 정리해주세요. (https://www.youtube.com/watch?v=7LVRciBEGaM 이 영상 참고하시면 이해가 쉬울것 같아서 올려보아요!)

---

### 3. Query / Key / Value 이해하기

Attention 안에는 아래 구조가 있습니다.

```
query:
현재 단어가 찾고 싶은 정보

key:
각 단어가 가진 특징

value:
실제로 참고할 정보
```

Query, Key, Value는 Attention이 단어 간 관계를 계산할 때 사용하는 요소입니다. 현재 단어가 문장 안의 다른 단어들 중 어떤 단어를 얼마나 참고할지 계산하는 과정이라고 이해할 수 있습니다.

즉, Query / Key / Value는 단어들이 서로를 참고하면서 문맥을 이해할 수 있도록 도와주는 구조입니다.

- [ ]  Query, Key, Value는 어떤 역할을 하나요?

---

### 4. Layer: 문장을 점점 깊게 이해하는 단계

모델 구조 출력 결과에서 아래 부분을 찾아보세요.

```
(0-11): 12 x BertLayer
```

이 뜻은 BERT 안에 Transformer Layer가 총 12개 있다는 의미입니다.

Transformer는 문장을 한 번만 읽고 끝내지 않습니다. 여러 layer를 거치며 문장을 점점 더 깊게 이해하는데요. 즉, Transformer는 layer를 반복적으로 거치며 문장의 의미를 점점 정교하게 분석합니다.

---

### 5. Hidden Size 이해하기

모델 구조에서 768이라는 숫자가 반복해서 등장합니다.

이 숫자는 하나의 토큰을 몇 개의 숫자로 표현하는지를 의미합니다. 

예를 들어 BERT에서 hidden size가 `768`이면,

`배`라는 토큰 하나를 모델 안에서는 숫자 768개짜리 벡터 형태로 표현한다는 뜻입니다. `배 → [0.12, -0.44, 0.91, ...]  총 768개`

그렇다면 왜 단어를 숫자 하나가 아니라 여러 개의 숫자로 표현할까요?

그 이유는 단어 하나에도 다양한 정보가 담겨야 하기 때문입니다.

예를 들어 단어에는, 의미, 문맥, 감정, 문장 안에서의 역할, 다른 단어와의 관계들같은 정보들이 포함될 수 있습니다. 

즉, 단어 하나에도 의미, 문맥, 감정, 문장 안에서의 역할 등 다양한 정보가 담길 수 있기 때문에, Transformer 모델은 단어를 숫자 하나가 아니라 hidden size 크기의 벡터로 표현합니다.

---

### 6. Pooler: 문장 전체 의미 만들기

모델 구조 출력 결과에서 아래 부분을 찾아보세요.

```
(pooler): BertPooler
```

지금까지는 저희가 단어 단위 정보 위주로 처리하는 단계를 설명드렸습니다. 하지만, 많은 작업은 문장 전체 의미가 필요합니다.

예를 들어, `배우 연기는 좋았지만 스토리는 지루했다` 이러한 리뷰가 있다고 가정해봅시다. 이 문장은 단어 하나만 보면 긍정인지 부정인지 판단하기 어렵습니다. 

따라서 Transformer는 여러 단어 정보를 종합하여 문장 전체 의미를 하나의 벡터로 정리하는 과정이 필요합니다.

이 역할을 하는 부분이 `Pooler`입니다.

- [ ]  Pooler의 역할을 정리해주세요.

---

### 7. Classifier: 최종 작업 수행하기

모델 구조 출력 결과에서 아래 부분을 찾아보세요.

```
(classifier): Linear(in_features=768, out_features=2)
```

지금까지는 Transformer 기반 모델이 문장을 어떻게 이해하는지를 설명드렸습니다. 이번 실습의 최종 목표는 긍정/부정을 분류하는 것입니다.

마지막 classifier가 이러한 분류 작업을 수행합니다. 

- Transformer 본체: 문장을 이해하는 역할
- classifier: 문제를 해결하는 역할

이라고 볼 수 있습니다.

# Mission2 (RAG)

Fine-Tuning은 모델 자체를 다시 학습시키는 방식이였습니다.

하지만 Fine-Tuning만으로는 해결하기 어려운 문제도 존재합니다.

사전학습 모델은 학습 당시의 데이터만 알고 있기 때문에, 최신 정보나 외부 문서를 자동으로 알 수는 없습니다.

이 문제를 해결하기 위해 등장한 방식이 바로 `RAG (Retrieval-Augmented Generation)`입니다. 

RAG란 질문과 관련된 외부 문서를 먼저 검색한 뒤, 그 문서를 함께 참고하여 답변하는 방식입니다. 

즉, 단순히 모델의 “기억”만 사용하는 것이 아니라, 검색(Retrieval) + 생성(Generation)을 함께 사용하는 구조입니다. 

## RAG 문장 처리 방식

```markdown
질문 입력
↓
질문을 벡터로 변환
↓
질문과 비슷한 문서 검색
↓
검색된 문서를 Context로 생성
↓
질문 + Context를 함께 모델에 입력
↓
최종 결과 생성
```

### **1. 문장 임베딩 모델과 벡터 검색 라이브러리 준비하기**

```python
!pip install sentence-transformers faiss-cpu
```

이번 실습에서는 두가지 라이브러리를 설치하도록 하겠습니다. 

- sentence-transformers: 문장을 벡터로 변환
- faiss: 벡터를 빠르게 검색

하기 위해 사용하는 라이브러리입니다.

- [ ]  `sentence-transformers` 라이브러리는 어떤 역할을 하는 라이브러리일까요?
- [ ]  `faiss`는 어떤 문제를 해결하기 위해 사용하는 라이브러리일까요?

### **2. RAG에 사용할 라이브러리 불러오기**

```python
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
```

### **3. Knowledge Base 만들기**

```python
knowledge_base = [
    "이 영화는 2023년 개봉한 한국 액션 블록버스터로 관객 800만 명을 돌파했다.",
    "주연 배우의 연기력이 뛰어나다는 평이 많으며 CG 퀄리티가 훌륭하다.",
    "스토리가 진부하고 예측 가능하다는 부정적인 의견도 존재한다.",
    "OST가 매우 감동적이며 엔딩 크레딧까지 자리를 뜨지 못하게 만든다.",
    "러닝타임이 2시간 30분으로 다소 길지만 지루하지 않다는 반응이다.",
    "아이맥스 상영관에서 보면 몰입감이 극대화된다는 관람객 후기가 많다."
]
```

RAG에서는 모델이 참고할 외부 문서가 필요합니다. 이러한 외부 문서 저장소를 
`Knowledge Base` 이라고 부릅니다. 즉, 모델이 학습한 정보만을 사용하는 것이 아니라, 필요한 정보를 외부에서 추가로 찾아 참고하는 구조입니다. 

- [ ]  Knowledge Base는 왜 필요한가요?

### **4. 문장을 벡터로 변환하기**

```python
embedder = SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2'
)

kb_embeddings = embedder.encode(
    knowledge_base,
    convert_to_numpy=True
)
```

RAG는 문장을 그대로 비교하지 않습니다.

대신 문장을 숫자 벡터로 변환한 뒤, 벡터 간 거리를 비교하여 의미가 비슷한 문서를 찾습니다.

예를 들어 `"배우 연기가 좋다"`와 `"주연 배우의 연기력이 뛰어나다"`는 글자가 완전히 같지는 않지만 의미는 비슷합니다.

문장 임베딩 모델은 이런 의미 정보를 벡터 공간에 반영합니다.

- [ ]  의미가 비슷한 문장은 벡터 공간에서 어떻게 나타날까요?

### **5. 벡터 검색 인덱스 만들기**

```python
dimension = kb_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(kb_embeddings)
```

RAG에서는 질문과 가장 비슷한 문서를 빠르게 찾아야 합니다.

문서가 몇 개 없다면 하나씩 비교해도 괜찮지만, 문서가 수천 개, 수만 개, 수백만 개로 많아지면 모든 문서를 직접 비교하는 방식은 너무 오래 걸립니다.

그래서 FAISS를 사용하여 문서 벡터들을 검색하기 좋은 형태로 저장합니다.

이때 만들어지는 검색용 구조를 `index`라고 부릅니다. 쉽게 말하면, `index`는 문서 벡터들을 빠르게 찾을 수 있도록 정리해둔 검색용 보관함입니다.

저희 코드에서는 L2 distance를 기준으로 비슷한 문서를 찾는 FAISS 검색기를 만드는 코드입니다.

- [ ]  FAISS를 사용하지 않고 직접 탐색하면 어떤 문제가 발생할까요?
- [ ]  L2 distance는 무엇을 의미하나요?

### **6. 질문과 비슷한 문서 검색하기**

```python
def retrieve(query, top_k=2):
    query_vec = embedder.encode(
        [query],
        convert_to_numpy=True
    )

    distances, indices = index.search(
        query_vec,
        top_k
    )

    return [knowledge_base[i] for i in indices[0]]
```

사용자의 질문과 가장 유사한 문서를 검색하는 함수입니다.

- **query → embedding**
    - 질문을 벡터로 변환
- **index.search**
    - 가장 가까운 문서 top_k개 검색
- **indices**
    - 유사한 문서의 인덱스 반환

즉, 질문을 벡터로 변환 → 문서들과 거리 계산 → 가장 가까운 문서 검색을 수행하는 함수인것이죠. 

- [ ]  retrieve 함수의 전체 흐름을 정리해주세요.
- [ ]  top_k 값은 어떤 역할을 하나요?

### **7. RAG 전체 파이프라인 이해하기**

```python
def rag_sentiment_pipeline(query):
    print(f"[질문]: {query}")

    retrieved = retrieve(query, top_k=2)
    context = " ".join(retrieved)
    print(f"[검색된 문서]: {context}")

    augmented_input = f"{context} {query}"
    result = fine_tuned_pipeline(augmented_input)[0]

    print(f"[감정 예측]: {format_result(result)}\n")
    print("-" * 60)
```

RAG의 핵심 파이프라인을 구성하는 단계입니다.

처음에 설명드렸던 RAG 파이프라인과 함께 설명드리자면, 

```markdown
질문 입력
→ query

↓
관련 문서 검색
→ retrieved = retrieve(query, top_k=2)

↓
검색된 문서를 Context로 생성
→ context = " ".join(retrieved)

↓
질문 + Context를 함께 모델에 입력
→ augmented_input = f"{context} {query}"

↓
최종 결과 생성
→ result = fine_tuned_pipeline(augmented_input)[0]
```

![image.png](attachment:42fbf63d-44c3-4869-9d26-ebfe2ba22823:image.png)

- [ ]  RAG 전체 구조를 정리해주세요. (위 그림을 참고하시면, 이해가 쉬울 것 같습니다.)

### **8. RAG 실행하기**

```python
queries = [
    "이 영화 배우 연기가 어때요?",
    "스토리는 어떤가요?",
    "아이맥스로 볼 만한가요?"
]

for q in queries:
    rag_sentiment_pipeline(q)
```

다양한 질문에 대해 RAG 파이프라인을 실행해봅시다.

질문과 관련된 문서를 먼저 검색한 뒤, 검색 결과를 바탕으로 모델이 답변하게 됩니다!

# TODO

- [ ]  각 코드 밑에 있는 미션을 블로그글로 정리해주세요.
- [ ]  300자 이상 WIL 작성하기 
- [ ]  코드를 보지 않고 실습하여 `.ipynb`로 저장하여, github에 올려주세요.

# 파일 구조

```
gdg-6th-ai-mission/
├── week1/
│   │   ├── week1_mission.ipynb 
│   │   └── [w](http://week1.md/)eek1.md
├── week2/
│   │   ├── week2_mission.ipynb 
│   │   └── [w](http://week2.md/)eek2.md
├── week3/
│   │   ├── week3_mission.ipynb # 실습 코드 파일
│   │   └── [w](http://week1.md/)eek3.md
│   │   
└── README.md

```