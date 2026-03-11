# 6장 B: 토픽 모델링 — 모범 답안과 해설

> 이 문서는 실습 제출 후 공개한다. 제출 전에는 열람하지 않는다.

---

## 실습 1 해설: LDA 토픽 모델링

### 제공 코드 실행 결과 해설

20 Newsgroups 데이터셋(4개 카테고리: 야구, 의료, 컴퓨터 그래픽, 정치)에서 2,000개 문서를 샘플링하여 LDA를 적용한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 최적 토픽 수 | K=4 (데이터의 실제 카테고리 수와 일치) | Coherence Score가 K=4에서 최고점 또는 엘보우 |
| C_v 일관성 | 0.55 전후 | 양호 수준. 토픽 내 키워드가 의미적으로 응집됨 |
| 토픽 다양성 | 0.72 전후 | 토픽 간 키워드 중복이 적음 |
| NPMI | 0.03 전후 | 상대적으로 낮음. 일부 일반적 단어(all, one 등) 포함 |

핵심 코드 구조:

```python
# CountVectorizer: LDA 입력을 위한 BoW 변환
# max_df=0.95: 거의 모든 문서에 등장하는 일반 단어 제거
# min_df=2: 최소 2개 문서에 등장하는 단어만 포함
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=5000)

# Gensim LDA 모델: alpha='auto'는 디리클레 파라미터를 자동 학습
lda_model = LdaModel(corpus=corpus, id2word=dictionary.id2token,
                      num_topics=4, random_state=42, passes=20, alpha='auto', eta='auto')
```

최적 K 선택 원리: Coherence Score가 높을수록 토픽 내 키워드 간 의미적 응집도가 높다. K를 너무 크게 하면 토픽이 세분화되어 중복 키워드가 증가하고, 너무 작으면 서로 다른 주제가 하나의 토픽으로 합쳐진다.

### 프롬프트 1 모범 구현: 토픽 수(K) 변화에 따른 평가지표 비교

```python
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

# 데이터 로드
categories = ['rec.sport.baseball', 'sci.med', 'comp.graphics', 'talk.politics.misc']
newsgroups = fetch_20newsgroups(subset='train', categories=categories,
                                remove=('headers', 'footers', 'quotes'), random_state=42)
indices = np.random.RandomState(42).choice(len(newsgroups.data), 2000, replace=False)
texts = [newsgroups.data[i] for i in indices]

# 전처리
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=5000)
doc_term_matrix = vectorizer.fit_transform(texts)
vocab = vectorizer.get_feature_names_out()
id2word = {i: word for i, word in enumerate(vocab)}

corpus = []
for i in range(doc_term_matrix.shape[0]):
    row = doc_term_matrix[i].toarray().flatten()
    doc_bow = [(j, int(row[j])) for j in range(len(row)) if row[j] > 0]
    corpus.append(doc_bow)

dictionary = corpora.Dictionary()
dictionary.id2token = id2word
dictionary.token2id = {v: k for k, v in id2word.items()}

tokenized_texts = [text.lower().split() for text in texts]

# K별 평가
print(f"{'K':<5} {'C_v':<10} {'UMass':<12} {'NPMI':<10} {'다양성':<10}")
print("-" * 47)

for k in range(2, 9):
    lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=k,
                    random_state=42, passes=15, alpha='auto', eta='auto')

    cv = CoherenceModel(model=lda, texts=tokenized_texts, dictionary=dictionary, coherence='c_v').get_coherence()
    umass = CoherenceModel(model=lda, corpus=corpus, dictionary=dictionary, coherence='u_mass').get_coherence()
    npmi = CoherenceModel(model=lda, texts=tokenized_texts, dictionary=dictionary, coherence='c_npmi').get_coherence()

    all_words = []
    for tid in range(k):
        words = lda.show_topic(tid, topn=10)
        all_words.extend([w for w, _ in words])
    diversity = len(set(all_words)) / len(all_words)

    print(f"{k:<5} {cv:<10.4f} {umass:<12.4f} {npmi:<10.4f} {diversity:<10.4f}")
```

기대 결과 해석:
- C_v는 K=4 근처에서 최고점 또는 엘보우를 보이는 경향. 실제 카테고리 수와 일치
- K가 커질수록 토픽 다양성은 처음에 증가하다가, K가 너무 크면 중복 키워드가 생겨 감소할 수 있음
- K=2는 너무 넓어서 해석이 모호하고, K=8은 너무 세분화되어 유사 토픽이 반복됨
- 핵심: 지표만으로 결정하지 말고, 키워드를 직접 읽어보며 해석 가능성을 최종 확인해야 함

### 프롬프트 2 모범 구현: LDA 토픽별 키워드 확률 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from gensim.models import LdaModel

# 데이터 및 모델 준비 (위와 동일)
categories = ['rec.sport.baseball', 'sci.med', 'comp.graphics', 'talk.politics.misc']
newsgroups = fetch_20newsgroups(subset='train', categories=categories,
                                remove=('headers', 'footers', 'quotes'), random_state=42)
indices = np.random.RandomState(42).choice(len(newsgroups.data), 2000, replace=False)
texts = [newsgroups.data[i] for i in indices]

vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=5000)
doc_term_matrix = vectorizer.fit_transform(texts)
vocab = vectorizer.get_feature_names_out()
id2word = {i: word for i, word in enumerate(vocab)}

corpus = []
for i in range(doc_term_matrix.shape[0]):
    row = doc_term_matrix[i].toarray().flatten()
    doc_bow = [(j, int(row[j])) for j in range(len(row)) if row[j] > 0]
    corpus.append(doc_bow)

dictionary = corpora.Dictionary()
dictionary.id2token = id2word
dictionary.token2id = {v: k for k, v in id2word.items()}

lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=4,
                random_state=42, passes=20, alpha='auto', eta='auto')

# 2x2 서브플롯
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for topic_id in range(4):
    topic_words = lda.show_topic(topic_id, topn=10)
    words = [w for w, _ in topic_words]
    probs = [p for _, p in topic_words]

    ax = axes[topic_id]
    y_pos = np.arange(len(words))
    ax.barh(y_pos, probs, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title(f'Topic {topic_id}')

plt.suptitle('LDA Topic Keywords and Probabilities', fontsize=14)
plt.tight_layout()
plt.savefig("lda_topic_keywords.png", dpi=150)
plt.show()
```

기대 결과: 각 토픽의 상위 키워드가 의미적으로 응집된 그룹을 보여준다. 예를 들어 야구 토픽은 "game", "team", "player" 등이, 의료 토픽은 "medical", "patient", "disease" 등이 상위에 위치한다.

---

## 실습 2 해설: BERTopic 토픽 모델링

### 제공 코드 실행 결과 해설

동일한 20 Newsgroups 데이터에 BERTopic을 적용한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 토픽 수 (축소 전) | 10~20개 이상 | HDBSCAN이 밀도 구조에 따라 자동 결정 |
| 토픽 수 (축소 후) | 5개 | `reduce_topics(nr_topics=5)`로 명시적 축소 |
| 노이즈(-1) 문서 | 100~300개 | HDBSCAN이 어떤 군집에도 속하지 않는 문서를 분리 |

핵심 코드 구조:

```python
# 5단계 파이프라인 명시적 구성
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")      # 384차원 임베딩
umap_model = UMAP(n_neighbors=15, n_components=5, ...)         # 384D → 5D
hdbscan_model = HDBSCAN(min_cluster_size=30, min_samples=10)   # 밀도 기반 군집화
vectorizer_model = CountVectorizer(stop_words='english')       # 토큰화
# c-TF-IDF는 BERTopic 내부에서 자동 수행
```

BERTopic이 LDA보다 문맥을 반영한 키워드를 추출하는 이유: 임베딩 단계에서 문맥 정보가 이미 반영되어, "bank(금융)" vs "bank(강가)"를 구분할 수 있다. 단어 빈도만 보는 LDA는 이 구분이 불가능하다.

reduce_outliers 적용 효과: 노이즈 문서의 대부분이 기존 토픽에 재할당된다. 특히 문서 수가 적은 토픽보다는 큰 토픽에 많이 할당되는데, 이는 더 많은 문서를 가진 토픽이 더 넓은 의미 범위를 커버하기 때문이다.

### 프롬프트 3 모범 구현: min_cluster_size 변경 실험

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# 데이터 로드
categories = ['rec.sport.baseball', 'sci.med', 'comp.graphics', 'talk.politics.misc']
newsgroups = fetch_20newsgroups(subset='train', categories=categories,
                                remove=('headers', 'footers', 'quotes'), random_state=42)
indices = np.random.RandomState(42).choice(len(newsgroups.data), 2000, replace=False)
texts = [newsgroups.data[i] for i in indices]
valid_idx = [i for i, t in enumerate(texts) if len(t.strip()) > 50]
texts = [texts[i] for i in valid_idx]

# 임베딩을 미리 계산 (반복 실험 효율화)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts, show_progress_bar=True)

print(f"문서 수: {len(texts)}")
print(f"\n{'min_cluster_size':<20} {'토픽 수':<10} {'노이즈 문서':<12} {'노이즈 비율':<12}")
print("-" * 54)

for mcs in [10, 20, 30, 50]:
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0,
                       metric='cosine', random_state=42)
    hdbscan_model = HDBSCAN(min_cluster_size=mcs, min_samples=10,
                             metric='euclidean', cluster_selection_method='eom')
    vectorizer_model = CountVectorizer(stop_words='english', min_df=2)

    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        verbose=False
    )

    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_noise = sum(1 for t in topics if t == -1)
    noise_ratio = n_noise / len(topics)

    print(f"{mcs:<20} {n_topics:<10} {n_noise:<12} {noise_ratio:<12.2%}")
```

기대 결과 해석:
- min_cluster_size=10: 토픽이 10개 이상으로 세분화되고, 노이즈 비율이 낮음
- min_cluster_size=30: 토픽이 5~8개로 적절하며, 노이즈 비율이 중간
- min_cluster_size=50: 토픽이 3~5개로 줄어들고, 노이즈 비율이 높아짐
- 핵심 트레이드오프: min_cluster_size가 크면 큰 토픽만 살아남아 해석이 쉽지만, 세부 주제를 놓칠 수 있음. 반대로 작으면 세부 주제를 포착하지만 해석 부담이 증가
- 실무에서는 먼저 작은 값으로 세분화된 토픽을 확인하고, `reduce_topics()`로 병합하여 원하는 수준으로 조정하는 것이 권장됨

### 프롬프트 4 모범 구현: LDA vs BERTopic 비교

```python
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from gensim import corpora
from gensim.models import LdaModel
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN

# 데이터 로드
categories = ['rec.sport.baseball', 'sci.med', 'comp.graphics', 'talk.politics.misc']
newsgroups = fetch_20newsgroups(subset='train', categories=categories,
                                remove=('headers', 'footers', 'quotes'), random_state=42)
indices = np.random.RandomState(42).choice(len(newsgroups.data), 2000, replace=False)
texts = [newsgroups.data[i] for i in indices]
labels = [newsgroups.target_names[newsgroups.target[i]] for i in indices]
valid_idx = [i for i, t in enumerate(texts) if len(t.strip()) > 50]
texts = [texts[i] for i in valid_idx]
labels = [labels[i] for i in valid_idx]

# --- LDA ---
vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english', max_features=5000)
doc_term_matrix = vectorizer.fit_transform(texts)
vocab = vectorizer.get_feature_names_out()
id2word = {i: word for i, word in enumerate(vocab)}
corpus = []
for i in range(doc_term_matrix.shape[0]):
    row = doc_term_matrix[i].toarray().flatten()
    doc_bow = [(j, int(row[j])) for j in range(len(row)) if row[j] > 0]
    corpus.append(doc_bow)
dictionary = corpora.Dictionary()
dictionary.id2token = id2word
dictionary.token2id = {v: k for k, v in id2word.items()}

lda = LdaModel(corpus=corpus, id2word=id2word, num_topics=4,
                random_state=42, passes=20, alpha='auto', eta='auto')

print("=" * 60)
print("LDA 토픽 키워드")
print("=" * 60)
for tid in range(4):
    words = [w for w, _ in lda.show_topic(tid, topn=7)]
    print(f"토픽 {tid}: {', '.join(words)}")

# --- BERTopic ---
topic_model = BERTopic(
    embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=30, min_samples=10, metric='euclidean', cluster_selection_method='eom'),
    vectorizer_model=CountVectorizer(stop_words='english', min_df=2),
    verbose=False
)
topics, _ = topic_model.fit_transform(texts)
topic_model.reduce_topics(texts, nr_topics=5)
topics = topic_model.topics_

print("\n" + "=" * 60)
print("BERTopic 토픽 키워드")
print("=" * 60)
for tid in topic_model.get_topic_info()['Topic']:
    if tid == -1:
        continue
    words = [w for w, _ in topic_model.get_topic(tid)[:7]]
    print(f"토픽 {tid}: {', '.join(words)}")

# BERTopic 토픽-레이블 매핑 정확도
df = pd.DataFrame({'topic': topics, 'label': labels})
print("\n" + "=" * 60)
print("BERTopic 토픽-레이블 매핑 정확도")
print("=" * 60)
for tid in sorted(df['topic'].unique()):
    if tid == -1:
        continue
    topic_docs = df[df['topic'] == tid]
    main_label = topic_docs['label'].value_counts().idxmax()
    accuracy = topic_docs['label'].value_counts().max() / len(topic_docs)
    print(f"토픽 {tid}: {main_label} (정확도: {accuracy:.2%}, {len(topic_docs)}건)")
```

기대 결과:
- BERTopic 토픽이 LDA보다 더 구체적이고 문맥 반영된 키워드를 보여줌
- BERTopic의 토픽-레이블 매핑 정확도가 80% 이상인 경우가 많음
- 특히 정치(politics) 토픽에서 LDA는 일반적 단어가 많이 섞이지만, BERTopic은 "government", "president" 등 핵심 단어가 상위에 위치

---

## 실습 3 해설: LLM 기반 토픽 레이블링

### 제공 코드 실행 결과 해설

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 키워드 패턴 매칭 | 4개 중 2~3개 정확 매핑 | 사전 정의된 패턴과 키워드가 부분 일치 |
| LLM 레이블 (API 키 있는 경우) | 4개 모두 의미 있는 이름 | GPT가 키워드 간 관계를 파악 |

핵심 코드 구조:

```python
# 키워드 패턴 기반 수동 레이블링의 한계
keyword_patterns = {
    ('game', 'team', 'player', 'baseball', 'hit'): '스포츠/야구',
    ...
}
# 패턴에 없는 키워드 조합이 나오면 → "토픽_N" (무의미)

# LLM 레이블링: 키워드 간 관계를 파악하여 적절한 이름 생성
representation_model = BertopicOpenAI(
    model="gpt-3.5-turbo",
    prompt="키워드를 바탕으로 토픽 이름을 5단어 이내로 지어주세요: [KEYWORDS]"
)
```

규칙 기반 vs LLM 레이블링의 차이: 규칙 기반은 사전에 정의한 패턴만 처리할 수 있어 확장성이 부족하다. LLM은 새로운 키워드 조합에 대해서도 맥락을 파악하여 적절한 이름을 생성한다.

### 프롬프트 5 모범 구현: 프롬프트 변형에 따른 레이블 품질 비교

```python
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# 데이터 로드
categories = ['rec.sport.baseball', 'sci.med', 'comp.graphics', 'talk.politics.misc']
newsgroups = fetch_20newsgroups(subset='train', categories=categories,
                                remove=('headers', 'footers', 'quotes'), random_state=42)
indices = np.random.RandomState(42).choice(len(newsgroups.data), 1500, replace=False)
texts = [newsgroups.data[i] for i in indices]
valid_idx = [i for i, t in enumerate(texts) if len(t.strip()) > 50]
texts = [texts[i] for i in valid_idx]

# BERTopic 기본 학습 (LLM 없이)
topic_model = BERTopic(
    embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
    verbose=False,
    calculate_probabilities=True
)
topics, probs = topic_model.fit_transform(texts)

# 토픽별 키워드 추출
topic_info = topic_model.get_topic_info()
topic_keywords = {}
for _, row in topic_info.iterrows():
    tid = row['Topic']
    if tid == -1:
        continue
    words = [w for w, _ in topic_model.get_topic(tid)[:5]]
    topic_keywords[tid] = words

# 3가지 레이블링 스타일 (규칙 기반 시뮬레이션)
def label_simple(keywords):
    """간단: 가장 대표적인 키워드 1개"""
    return keywords[0] if keywords else "unknown"

def label_detailed(keywords):
    """상세: 키워드 패턴으로 카테고리 추론"""
    sports = {'game', 'team', 'player', 'baseball', 'hit', 'run', 'season', 'win'}
    medical = {'medical', 'doctor', 'patient', 'health', 'disease', 'treatment', 'drug'}
    tech = {'graphics', 'image', 'file', 'software', 'computer', 'program', 'data'}
    politics = {'government', 'president', 'state', 'law', 'political', 'people', 'policy'}

    kw_set = set(keywords)
    scores = {
        '스포츠/야구 (confidence: 0.8)': len(kw_set & sports),
        '의료/건강 (confidence: 0.8)': len(kw_set & medical),
        '컴퓨터/그래픽 (confidence: 0.8)': len(kw_set & tech),
        '정치/정부 (confidence: 0.8)': len(kw_set & politics),
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else f"미분류 ({', '.join(keywords[:2])})"

def label_domain(keywords):
    """도메인: 뉴스 섹션 매핑"""
    sports = {'game', 'team', 'player', 'baseball', 'hit', 'run', 'season'}
    medical = {'medical', 'doctor', 'patient', 'health', 'disease', 'treatment'}
    tech = {'graphics', 'image', 'file', 'software', 'computer', 'program'}
    politics = {'government', 'president', 'state', 'law', 'political', 'people'}

    kw_set = set(keywords)
    sections = {
        '[스포츠면] 야구/스포츠 뉴스': len(kw_set & sports),
        '[사회면] 의료/건강 뉴스': len(kw_set & medical),
        '[IT면] 컴퓨터/기술 뉴스': len(kw_set & tech),
        '[정치면] 정치/정부 뉴스': len(kw_set & politics),
    }
    best = max(sections, key=sections.get)
    return best if sections[best] > 0 else f"[기타] {keywords[0]} 관련"

# 비교 출력
print(f"{'토픽':<8} {'키워드':<40} {'간단':<15} {'상세':<35} {'도메인':<30}")
print("-" * 128)
for tid, kws in sorted(topic_keywords.items()):
    kw_str = ', '.join(kws)
    simple = label_simple(kws)
    detailed = label_detailed(kws)
    domain = label_domain(kws)
    print(f"{tid:<8} {kw_str:<40} {simple:<15} {detailed:<35} {domain:<30}")

# OpenAI API가 있는 경우 실제 LLM 레이블링 (선택)
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    from openai import OpenAI
    client = OpenAI()

    prompts = {
        '간단': "다음 키워드를 한 단어로 요약해줘: {keywords}",
        '상세': "다음 키워드를 분석하여 5단어 이내 토픽명을 JSON으로 반환해줘: {keywords}",
        '도메인': "당신은 뉴스 기사 분류 전문가입니다. 다음 키워드가 어떤 뉴스 섹션에 해당하는지 판단해줘: {keywords}"
    }

    print("\n\n[LLM 레이블링 결과]")
    for tid, kws in sorted(topic_keywords.items()):
        kw_str = ', '.join(kws)
        print(f"\n토픽 {tid}: {kw_str}")
        for style, prompt_template in prompts.items():
            prompt = prompt_template.format(keywords=kw_str)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            label = response.choices[0].message.content.strip()
            print(f"  {style}: {label}")
```

기대 결과 해석:
- 간단 스타일: "game", "medical" 등 단일 키워드 반환. 빠르지만 구체성 부족
- 상세 스타일: "Baseball & Sports Discussion" 등 맥락을 반영한 구체적 이름
- 도메인 스타일: "[스포츠면] 야구 관련 뉴스" 등 비즈니스에 즉시 활용 가능한 분류
- 핵심: 실무에서는 도메인 스타일이 가장 유용. 분석 목적에 맞는 추상화 수준을 프롬프트로 조절

---

## 실습 4 해설: BERTopic + 동적 토픽 분석

### 제공 코드 실행 결과 해설

한국어 뉴스 시뮬레이션 데이터(2019~2021, 1,000개 문서)에 Kiwi 명사 추출 + BERTopic을 적용한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 정적 토픽 수 | 5~10개 | 데이터의 실제 주제(경제, 코로나, 기술, 재택근무) + 세부 분화 |
| 코로나 토픽 출현 | 2019-12~2020-01 | 팬데믹 시작 시점 반영 |
| 화상회의 토픽 최고점 | 2020-09 전후 | 거리두기 강화 시기와 일치 |

핵심 코드 구조:

```python
# Kiwi 명사 추출: 조사/어미 제거로 토픽 해석성 향상
target_tags = {'NNG', 'NNP', 'SL'}  # 일반명사, 고유명사, 알파벳
nouns = [token.form for token in result[0][0] if token.tag in target_tags and len(token.form) > 1]

# 동적 토픽 분석
topics_over_time = topic_model.topics_over_time(texts, timestamps, nr_bins=12)

# 토픽 생명 주기: 출현/최고점/소멸 자동 식별
topic_lifecycle = analyze_topic_lifecycle(topics_over_time, topic_model, threshold=0.02)
```

명사 추출의 효과: "코로나19 확진자가 급증하며"에서 명사만 추출하면 "코로나 확진자 급증"이 되어, "가", "며" 같은 조사가 키워드에 포함되지 않는다.

### 프롬프트 6 모범 구현: nr_bins 변경에 따른 동적 분석 비교

```python
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from kiwipiepy import Kiwi

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# 데이터 로드
CURRENT_DIR = os.path.dirname(__file__) if '__file__' in dir() else '.'
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data')
csv_path = os.path.join(DATA_DIR, 'korean_news.csv')
df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])

# 명사 추출
kiwi = Kiwi()
target_tags = {'NNG', 'NNP', 'SL'}
texts_nouns = []
for text in df['text']:
    result = kiwi.analyze(text)
    nouns = [token.form for token in result[0][0] if token.tag in target_tags and len(token.form) > 1]
    texts_nouns.append(" ".join(nouns))

timestamps = df['date'].tolist()

# BERTopic 학습
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts_nouns, show_progress_bar=True)

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=15, min_samples=10, metric='euclidean',
                           cluster_selection_method='eom', prediction_data=True),
    verbose=False
)
topics, probs = topic_model.fit_transform(texts_nouns, embeddings=embeddings)

# 상위 3개 토픽
topic_info = topic_model.get_topic_info()
top_topics = [t for t in topic_info['Topic'] if t != -1][:3]

# nr_bins별 비교 시각화
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
bins_list = [6, 12, 24, 36]

for idx, (nr_bins, ax) in enumerate(zip(bins_list, axes.flatten())):
    tot = topic_model.topics_over_time(texts_nouns, timestamps, nr_bins=nr_bins)

    for topic in top_topics:
        topic_data = tot[tot['Topic'] == topic].sort_values('Timestamp')
        keywords = [w for w, _ in topic_model.get_topic(topic)[:3]]
        label = f"토픽 {topic}: {', '.join(keywords)}"
        ax.plot(topic_data['Timestamp'], topic_data['Frequency'], label=label, marker='o', markersize=3)

    ax.set_title(f'nr_bins={nr_bins}', fontsize=12)
    ax.set_xlabel('시간')
    ax.set_ylabel('빈도')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('nr_bins 변경에 따른 동적 토픽 분석 비교', fontsize=14)
plt.tight_layout()
plt.savefig("nr_bins_comparison.png", dpi=150)
plt.show()
```

기대 결과 해석:
- nr_bins=6 (반년별): 전체적인 추세만 보임. 세부 변동을 놓칠 수 있음
- nr_bins=12 (분기별): 주요 추세와 계절성을 균형 있게 포착. 가장 실용적
- nr_bins=24 (~1.5개월): 세밀한 변동이 보이지만 노이즈도 증가
- nr_bins=36 (월별): 매우 세밀하지만 각 구간의 문서 수가 적어 빈도가 들쭉날쭉
- 핵심: 분석 목적에 따라 선택. 전략 보고서에는 nr_bins=12, 위기 모니터링에는 nr_bins=36이 적합

### 프롬프트 7 모범 구현: 토픽 간 시계열 상관관계 분석

```python
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from kiwipiepy import Kiwi

# 한글 폰트 설정
import platform
if platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
elif platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# 데이터 로드 및 전처리 (위와 동일)
CURRENT_DIR = os.path.dirname(__file__) if '__file__' in dir() else '.'
DATA_DIR = os.path.join(CURRENT_DIR, '..', 'data')
df = pd.read_csv(os.path.join(DATA_DIR, 'korean_news.csv'))
df['date'] = pd.to_datetime(df['date'])

kiwi = Kiwi()
target_tags = {'NNG', 'NNP', 'SL'}
texts_nouns = []
for text in df['text']:
    result = kiwi.analyze(text)
    nouns = [token.form for token in result[0][0] if token.tag in target_tags and len(token.form) > 1]
    texts_nouns.append(" ".join(nouns))
timestamps = df['date'].tolist()

# BERTopic 학습
topic_model = BERTopic(
    embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
    umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42),
    hdbscan_model=HDBSCAN(min_cluster_size=15, min_samples=10, prediction_data=True),
    verbose=False
)
topics, _ = topic_model.fit_transform(texts_nouns)

# 동적 토픽 분석
topics_over_time = topic_model.topics_over_time(texts_nouns, timestamps, nr_bins=12)

# 피벗하여 토픽별 시계열 생성
pivot_df = topics_over_time.pivot(index='Timestamp', columns='Topic', values='Frequency').fillna(0)

# 노이즈(-1) 제거
if -1 in pivot_df.columns:
    pivot_df = pivot_df.drop(columns=[-1])

# 상관관계 행렬
corr_matrix = pivot_df.corr()

# 토픽 이름으로 레이블 변경
topic_labels = {}
for col in pivot_df.columns:
    keywords = [w for w, _ in topic_model.get_topic(col)[:3]]
    topic_labels[col] = f"T{col}: {', '.join(keywords)}"

corr_matrix_labeled = corr_matrix.rename(index=topic_labels, columns=topic_labels)

# 히트맵 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix_labeled, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True)
plt.title('토픽 간 시계열 상관관계', fontsize=14)
plt.tight_layout()
plt.savefig("topic_correlation_heatmap.png", dpi=150)
plt.show()

# 높은 상관관계 쌍 출력
print("\n상관관계가 높은 토픽 쌍 (|r| > 0.5):")
print("-" * 60)
for i in range(len(corr_matrix)):
    for j in range(i + 1, len(corr_matrix)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.5:
            t1 = corr_matrix.index[i]
            t2 = corr_matrix.columns[j]
            kw1 = ', '.join([w for w, _ in topic_model.get_topic(t1)[:3]])
            kw2 = ', '.join([w for w, _ in topic_model.get_topic(t2)[:3]])
            direction = "양의" if r > 0 else "음의"
            print(f"토픽 {t1}({kw1}) ↔ 토픽 {t2}({kw2}): r={r:.3f} ({direction} 상관)")
```

기대 결과 해석:
- 코로나(확진)와 화상회의/재택근무는 양의 상관관계가 높을 것: 팬데믹이 원격 근무를 촉발
- 경제(주식)와 코로나는 음의 상관관계가 있을 수 있음: 팬데믹 초기 주식 시장 관심 감소
- 핵심: 상관관계는 인과관계가 아님. 두 토픽이 같은 시기에 빈도가 함께 변했다는 것을 의미할 뿐

---

## 실습 5 해설: 멀티모달 토픽 모델링

### 제공 코드 실행 결과 해설

패션 상품 데이터(20개, 드레스/재킷/신발/가방)에 CLIP 기반 멀티모달 분석을 적용한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 토픽 수 | 3개 (+노이즈 1개) | 20개 샘플에서 min_topic_size=3으로 소규모 토픽 허용 |
| 드레스+재킷 → 같은 토픽 | Topic 0 (의류) | CLIP이 "착용하는 의류"라는 시각적 공통점 포착 |
| 신발 → 별도 토픽 | Topic 1 (신발) | 형태적 차이가 명확 |
| 가방 → 별도 토픽 | Topic 2 (가방) | 형태적 차이가 명확 |

핵심 코드 구조:

```python
# CLIP 모델: 텍스트와 이미지를 동일한 512차원 공간에 투영
embedding_model = MultiModalBackend('clip-ViT-B-32', batch_size=32)

# 텍스트 캡션과 이미지 경로를 함께 학습
topics, probs = topic_model.fit_transform(documents=captions, images=image_paths)
```

멀티모달의 장점: 텍스트만으로는 "leather"가 재킷/신발/가방에 공통 등장하여 혼동되지만, 이미지를 결합하면 시각적 형태 차이(의류 vs 신발 vs 가방)로 정확히 구분된다.

### 프롬프트 8 모범 구현: Unimodal vs Multimodal 비교

```python
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from bertopic import BERTopic
from bertopic.backend import MultiModalBackend
from bertopic.representation import VisualRepresentation
from sentence_transformers import SentenceTransformer

# 데이터 로드
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_PATH, "data")
df = pd.read_csv(os.path.join(DATA_PATH, "multimodal_data.csv"))
captions = df['caption'].tolist()
image_paths = [os.path.join(DATA_PATH, path) for path in df['image_path']]

print(f"데이터: {len(captions)}개 문서/이미지")

# --- (1) 텍스트 전용 BERTopic ---
print("\n" + "=" * 60)
print("(1) 텍스트 전용 BERTopic (all-MiniLM-L6-v2)")
print("=" * 60)

text_model = BERTopic(
    embedding_model=SentenceTransformer("all-MiniLM-L6-v2"),
    verbose=False,
    min_topic_size=3
)
text_topics, _ = text_model.fit_transform(captions)

text_info = text_model.get_topic_info()
text_n_topics = len(text_info[text_info['Topic'] != -1])
text_noise = sum(1 for t in text_topics if t == -1)
text_noise_ratio = text_noise / len(text_topics)

print(f"토픽 수: {text_n_topics}")
print(f"노이즈: {text_noise}개 ({text_noise_ratio:.0%})")
for _, row in text_info.iterrows():
    tid = row['Topic']
    if tid == -1:
        continue
    words = [w for w, _ in text_model.get_topic(tid)[:5]]
    print(f"  토픽 {tid}: {', '.join(words)} ({row['Count']}건)")

# --- (2) 멀티모달 BERTopic (CLIP) ---
print("\n" + "=" * 60)
print("(2) 멀티모달 BERTopic (CLIP ViT-B-32)")
print("=" * 60)

mm_embedding = MultiModalBackend('clip-ViT-B-32', batch_size=32)
visual_model = VisualRepresentation()

mm_model = BERTopic(
    embedding_model=mm_embedding,
    representation_model={"Visual_Aspect": visual_model},
    verbose=False,
    min_topic_size=3
)
mm_topics, _ = mm_model.fit_transform(documents=captions, images=image_paths)

mm_info = mm_model.get_topic_info()
mm_n_topics = len(mm_info[mm_info['Topic'] != -1])
mm_noise = sum(1 for t in mm_topics if t == -1)
mm_noise_ratio = mm_noise / len(mm_topics)

print(f"토픽 수: {mm_n_topics}")
print(f"노이즈: {mm_noise}개 ({mm_noise_ratio:.0%})")
for _, row in mm_info.iterrows():
    tid = row['Topic']
    if tid == -1:
        continue
    words = [w for w, _ in mm_model.get_topic(tid)[:5]]
    print(f"  토픽 {tid}: {', '.join(words)} ({row['Count']}건)")

# --- 비교 요약 ---
print("\n" + "=" * 60)
print("비교 요약")
print("=" * 60)
print(f"{'분석 방식':<25} {'토픽 수':<10} {'노이즈 비율':<12} {'특징'}")
print("-" * 70)
print(f"{'텍스트 단독':<25} {text_n_topics:<10} {text_noise_ratio:<12.0%} {'동의어 혼동 가능'}")
print(f"{'멀티모달 (CLIP)':<25} {mm_n_topics:<10} {mm_noise_ratio:<12.0%} {'시각+의미 결합'}")
```

기대 결과 해석:
- 텍스트 단독: "leather" 키워드가 재킷/신발/가방에 공통 등장하여 토픽이 혼동될 수 있음. 노이즈 비율이 상대적으로 높을 수 있음
- 멀티모달 (CLIP): 이미지의 시각적 특징(의류 형태 vs 신발 형태 vs 가방 형태)이 임베딩에 반영되어, 텍스트만으로 구분 어려운 카테고리를 정확히 분리
- 20개 소규모 데이터에서도 멀티모달의 장점이 나타남: 텍스트만으로는 부족한 정보를 이미지가 보완
- 핵심: 멀티모달 분석은 텍스트와 이미지가 상호 보완적일 때 가장 효과적. 텍스트만으로 충분한 경우 추가적인 계산 비용이 정당화되지 않을 수 있음

---

## 6장 전체 핵심 정리

```text
1. LDA는 단어 빈도 기반의 생성 모델로, alpha 파라미터가 토픽 혼합 희소성을 조절한다.
   K를 결정할 때는 Coherence Score와 토픽 키워드의 해석 가능성을 함께 판단한다.
2. BERTopic의 5단계 파이프라인은 각 모듈이 독립적으로 교체 가능하여 유연하다.
   min_cluster_size가 토픽 세분화 수준을 조절하는 핵심 파라미터다.
3. LLM 레이블링의 품질은 프롬프트에 크게 의존한다. 도메인 컨텍스트와
   Few-shot 예시를 제공하면 비즈니스에 즉시 활용 가능한 레이블을 얻을 수 있다.
4. 동적 토픽 분석에서 nr_bins는 시간 해상도를 결정한다. 너무 크면 통계적 불안정,
   너무 작으면 세부 변동을 놓친다. 분석 목적에 맞게 선택한다.
5. 토픽 간 상관관계 분석은 함께 변동하는 주제들을 발견한다.
   상관관계는 인과관계가 아니므로 해석에 주의가 필요하다.
6. 멀티모달 분석은 텍스트와 이미지를 CLIP으로 동일 공간에 투영하여,
   단일 모달리티보다 정확한 토픽 분류가 가능하다.
7. AI 도구로 코드를 생성하되, 결과를 반드시 검증하고 해석하는 습관이 핵심이다.
```
