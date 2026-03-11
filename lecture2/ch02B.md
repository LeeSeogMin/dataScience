# 2장 C: 데이터 전처리와 특성 공학 — 모범 답안과 해설

> 이 문서는 B회차 실습 제출 후 공개한다. 제출 전에는 열람하지 않는다.

---

## 실습 1 해설: 결측치 처리 비교

### 제공 코드 실행 결과 해설

California Housing 데이터(20,640샘플, 8특성)에 20% 결측을 도입한 결과:

| 방법 | RMSE 경향 | 이유 |
| ---- | --------- | ---- |
| 행 삭제 | 가장 높음 | 1-(0.8)⁸ ≈ 83% 행이 삭제됨. 훈련 데이터가 약 17%로 줄어 학습 부족 |
| 평균 대체 | 중간 | 분산이 줄어 세부 패턴 반영 약함 |
| 중앙값 대체 | 중간 | 평균과 유사. 이상치에 약간 더 강건 |
| KNN 대체 | 낮음 | 가까운 5개 샘플의 값을 활용해 원래 값에 더 가까운 대체 가능 |
| MICE | 낮음 | 변수 간 상관관계를 반복적으로 활용 |

핵심 코드 구조:

```python
# 파이프라인: 대체 → 스케일링 → 모델을 묶어 누수 방지
pipeline = Pipeline([
    ('imputer', imputer),
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0))
])
```

행 삭제만 파이프라인 밖에서 처리하는 이유: NaN 행을 먼저 제거해야 파이프라인에 넣을 수 있기 때문이다.

### 프롬프트 1 모범 구현: 결측 비율 변경 실험

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

missing_rates = [0.1, 0.2, 0.3, 0.5]
methods = [
    ("행 삭제", None),
    ("평균 대체", SimpleImputer(strategy='mean')),
    ("KNN 대체", KNNImputer(n_neighbors=5)),
]

print(f"{'결측비율':<10}", end="")
for name, _ in methods:
    print(f"{name:<15}", end="")
print()

for rate in missing_rates:
    np.random.seed(42)
    mask_train = np.random.random(X_train.shape) < rate
    mask_test = np.random.random(X_test.shape) < rate
    X_tr = X_train.copy(); X_tr[mask_train] = np.nan
    X_te = X_test.copy(); X_te[mask_test] = np.nan

    print(f"{rate:<10.0%}", end="")
    for name, imp in methods:
        if name == "행 삭제":
            tr_ok = ~np.isnan(X_tr).any(axis=1)
            te_ok = ~np.isnan(X_te).any(axis=1)
            pipe = Pipeline([('s', StandardScaler()), ('m', Ridge())])
            pipe.fit(X_tr[tr_ok], y_train[tr_ok])
            rmse = np.sqrt(mean_squared_error(y_test[te_ok], pipe.predict(X_te[te_ok])))
        else:
            pipe = Pipeline([('i', imp), ('s', StandardScaler()), ('m', Ridge())])
            pipe.fit(X_tr, y_train)
            rmse = np.sqrt(mean_squared_error(y_test, pipe.predict(X_te)))
        print(f"{rmse:<15.3f}", end="")
    print()
```

기대 결과 해석:
- 결측 비율이 올라갈수록 행 삭제의 RMSE가 급격히 나빠진다 (50%에서는 거의 모든 데이터 삭제)
- KNN 대체가 전 구간에서 가장 안정적이다
- 평균 대체는 비율이 높아지면 분산 왜곡이 심해져 성능이 떨어진다

### 프롬프트 2 모범 구현: 결측 지시 변수

```python
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer, MissingIndicator

# 평균 대체만
pipe_simple = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('model', Ridge())
])

# 평균 대체 + 결측 지시 변수
pipe_indicator = Pipeline([
    ('features', FeatureUnion([
        ('imputed', Pipeline([
            ('imp', SimpleImputer(strategy='mean')),
            ('scale', StandardScaler())
        ])),
        ('indicator', MissingIndicator())
    ])),
    ('model', Ridge())
])

pipe_simple.fit(X_tr, y_train)
pipe_indicator.fit(X_tr, y_train)

rmse_simple = np.sqrt(mean_squared_error(y_test, pipe_simple.predict(X_te)))
rmse_indicator = np.sqrt(mean_squared_error(y_test, pipe_indicator.predict(X_te)))

print(f"평균 대체만: RMSE = {rmse_simple:.3f}")
print(f"평균 대체 + 지시변수: RMSE = {rmse_indicator:.3f}")
```

기대 결과: 지시 변수를 추가하면 RMSE가 약간 개선될 수 있다. 결측 여부 자체가 예측에 유용한 정보를 담고 있기 때문이다.

---

## 실습 2 해설: 스케일링과 파이프라인

### 제공 코드 실행 결과 해설

- KNN: StandardScaler 적용 시 RMSE가 수 %p 개선. 스케일링 없으면 Population(0~35,000)이 MedInc(0~15)를 압도
- Random Forest: 스케일링 유무와 무관. 분할 기준은 값의 순서에만 의존
- 데이터 누수: 이 데이터에서 차이는 작지만, 시계열이나 concept drift가 있는 환경에서는 큰 차이 가능

### 프롬프트 3 모범 구현: 이상치와 스케일링

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 이상치가 없는 경우
results = {}
for name, scaler in [("Standard", StandardScaler()), ("Robust", RobustScaler())]:
    pipe = Pipeline([('s', scaler), ('m', KNeighborsRegressor(n_neighbors=5))])
    pipe.fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, pipe.predict(X_test)))
    results[f"정상_{name}"] = rmse

# 이상치 추가: 상위 1% 값을 100배
X_train_out = X_train.copy()
for col in range(X_train_out.shape[1]):
    threshold = np.percentile(X_train_out[:, col], 99)
    X_train_out[X_train_out[:, col] > threshold, col] *= 100

for name, scaler in [("Standard", StandardScaler()), ("Robust", RobustScaler())]:
    pipe = Pipeline([('s', scaler), ('m', KNeighborsRegressor(n_neighbors=5))])
    pipe.fit(X_train_out, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, pipe.predict(X_test)))
    results[f"이상치_{name}"] = rmse

print(f"{'조건':<25} {'StandardScaler':<18} {'RobustScaler':<18}")
print(f"{'이상치 없음':<25} {results['정상_Standard']:<18.4f} {results['정상_Robust']:<18.4f}")
print(f"{'이상치 있음':<25} {results['이상치_Standard']:<18.4f} {results['이상치_Robust']:<18.4f}")
```

기대 결과:
- 이상치 없음: 두 스케일러 차이 작음
- 이상치 있음: StandardScaler의 RMSE가 크게 나빠짐. 이상치가 평균과 표준편차를 왜곡하기 때문
- RobustScaler는 중앙값/IQR을 사용하므로 이상치에 영향을 거의 받지 않음

### 프롬프트 4 모범 구현: 교차검증

```python
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsRegressor(n_neighbors=5))
])

scores = cross_val_score(pipe, X_train, y_train,
                         cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-scores)

print(f"5-fold RMSE: {rmse_scores}")
print(f"평균 RMSE: {rmse_scores.mean():.4f} ± {rmse_scores.std():.4f}")
```

핵심: `cross_val_score`는 파이프라인을 받으면 각 폴드마다 `scaler.fit_transform(X_train_fold)` → `scaler.transform(X_val_fold)`를 자동 수행한다. 이것이 파이프라인을 쓰는 가장 중요한 이유다.

---

## 실습 3 해설: 차원 축소 비교

### 제공 코드 실행 결과 해설

| 방법 | 처리 시간 | Silhouette Score | 특징 |
| ---- | --------- | ---------------- | ---- |
| PCA | 0.01초 이하 | 0.1~0.2 | 선형, 빠름, 군집 겹침 |
| t-SNE | 수 초~수십 초 | 0.4~0.6 | 비선형, 느림, 군집 선명 |
| UMAP | 수 초 | 0.4~0.6 | 비선형, 중간 속도, 군집 선명 |

PCA의 PC1+PC2는 전체 분산의 약 20~30%만 설명한다. 90% 설명에 약 20개, 95%에 약 30개 주성분이 필요하다. → 64차원 데이터를 2차원으로 줄이면 정보 손실이 크지만, 대략적인 구조 파악은 가능하다.

### 프롬프트 5 모범 구현: perplexity 실험

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
perplexities = [5, 15, 30, 50]

for ax, perp in zip(axes.flatten(), perplexities):
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000)
    X_2d = tsne.fit_transform(X)
    sil = silhouette_score(X_2d, y)
    ax.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='tab10', s=5, alpha=0.7)
    ax.set_title(f"perplexity={perp}, Silhouette={sil:.3f}")

plt.tight_layout()
plt.savefig("tsne_perplexity_comparison.png", dpi=150)
plt.show()
```

기대 결과:
- perplexity=5: 매우 작은 군집이 많이 생김 (과도하게 지역적)
- perplexity=30: 보통 가장 균형 잡힌 결과
- perplexity=50: 군집이 뭉개지기 시작
- 값에 따라 Silhouette Score가 크게 변한다 → t-SNE 결과를 하나만 보고 판단하면 안 된다

### 프롬프트 6 모범 구현: PCA 전처리 후 분류

```python
import time
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

digits = load_digits()
X = StandardScaler().fit_transform(digits.data)
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dims = [64, 20, 10]
for d in dims:
    start = time.time()
    if d < 64:
        pca = PCA(n_components=d)
        X_tr = pca.fit_transform(X_train)
        X_te = pca.transform(X_test)
    else:
        X_tr, X_te = X_train, X_test

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_train)
    acc = accuracy_score(y_test, knn.predict(X_te))
    elapsed = time.time() - start
    print(f"차원={d:<4} 정확도={acc:.4f}  시간={elapsed:.3f}초")
```

기대 결과:
- 64→20차원: 정확도 거의 동일 또는 약간 개선 (노이즈 제거 효과)
- 64→10차원: 정확도가 약간 떨어질 수 있음 (정보 손실 시작)
- 처리 시간: 차원이 줄수록 KNN의 거리 계산이 빨라짐
- 결론: 적절한 차원 축소는 성능을 유지하면서 속도를 높인다

---

## 실습 4 해설: TF-IDF vs 임베딩 비교

### 제공 코드 실행 결과 해설

| 방법 | 정확도 | 벡터 차원 | 핵심 |
| ---- | ------ | --------- | ---- |
| TF-IDF | 0.80~0.85 | 5,000 | 빠르지만 의미 반영 안 됨. 대부분 0인 희소 벡터 |
| SBERT | 0.85~0.90 | 384 | 느리지만 의미 반영. 모든 차원에 정보 밀집 |

실험 설계의 핵심: 분류기를 동일(로지스틱 회귀)하게 고정하고 입력만 바꿨으므로, 정확도 차이는 전적으로 텍스트 표현 방법의 차이에서 온다.

### 프롬프트 7 모범 구현: 코사인 유사도 비교

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

sentences = [
    "The movie was great and I enjoyed it",
    "I really liked this excellent film",
    "The weather is sunny today"
]

# TF-IDF
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(sentences).toarray()
sim_tfidf = cosine_similarity(X_tfidf)

# SBERT
model = SentenceTransformer('all-MiniLM-L6-v2')
X_sbert = model.encode(sentences)
sim_sbert = cosine_similarity(X_sbert)

pairs = [(0,1,"영화1-영화2"), (0,2,"영화1-날씨"), (1,2,"영화2-날씨")]
print(f"{'문장 쌍':<15} {'TF-IDF':<12} {'SBERT':<12}")
for i, j, name in pairs:
    print(f"{name:<15} {sim_tfidf[i,j]:<12.3f} {sim_sbert[i,j]:<12.3f}")
```

기대 결과:
- TF-IDF: 문장1-문장2 유사도가 낮음. "great"과 "excellent", "movie"와 "film"이 겹치지 않기 때문
- SBERT: 문장1-문장2 유사도가 높음. 의미적으로 비슷한 문장임을 벡터 거리로 반영
- 두 방법 모두 영화-날씨 쌍의 유사도는 낮음
- 이것이 "단어 개수표 vs 의미 지도"의 핵심 차이다

### 프롬프트 8 모범 구현: 샘플 수와 성능

```python
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sentence_transformers import SentenceTransformer

categories = ['comp.graphics', 'sci.med', 'rec.sport.baseball', 'talk.politics.misc']
news = fetch_20newsgroups(subset='all', categories=categories,
                          remove=('headers','footers','quotes'), random_state=42)

texts_train, texts_test, y_train, y_test = train_test_split(
    news.data, news.target, test_size=0.2, random_state=42, stratify=news.target)

sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

train_sizes = [100, 500, 1000, len(texts_train)]
print(f"{'훈련 수':<10} {'TF-IDF':<12} {'SBERT':<12} {'차이':<10}")
for n in train_sizes:
    tr_texts = texts_train[:n]
    tr_y = y_train[:n]

    # TF-IDF
    vec = TfidfVectorizer(max_features=5000, stop_words='english')
    X_tr_tf = vec.fit_transform(tr_texts).toarray()
    X_te_tf = vec.transform(texts_test).toarray()
    clf_tf = LogisticRegression(max_iter=1000, random_state=42)
    clf_tf.fit(X_tr_tf, tr_y)
    acc_tf = accuracy_score(y_test, clf_tf.predict(X_te_tf))

    # SBERT
    X_tr_sb = sbert_model.encode(tr_texts)
    X_te_sb = sbert_model.encode(texts_test)
    clf_sb = LogisticRegression(max_iter=1000, random_state=42)
    clf_sb.fit(X_tr_sb, tr_y)
    acc_sb = accuracy_score(y_test, clf_sb.predict(X_te_sb))

    print(f"{n:<10} {acc_tf:<12.3f} {acc_sb:<12.3f} {(acc_sb-acc_tf)*100:+.1f}%p")
```

기대 결과:
- 훈련 100개: SBERT가 TF-IDF보다 10~15%p 이상 높음
- 훈련 1000개 이상: 차이가 줄어들지만 여전히 SBERT가 우세
- 이유: SBERT는 사전학습된 언어 지식을 이미 갖고 있어 적은 데이터에서도 의미를 파악 가능. TF-IDF는 데이터가 적으면 어휘 부족으로 표현력이 떨어짐

---

## 2장 전체 핵심 정리

```text
1. 결측치: 비율이 높아지면 행 삭제는 사실상 불가. KNN/MICE가 안정적이지만 비용이 큼.
   결측 지시 변수로 결측 자체를 정보로 활용할 수 있다.
2. 스케일링: KNN은 스케일링 필수, 트리 모델은 불필요. 이상치가 많으면 RobustScaler.
   파이프라인이 교차검증에서 누수를 자동 방지하는 원리를 이해해야 한다.
3. 차원 축소: t-SNE는 perplexity에 민감하므로 하나의 결과만 보고 판단하면 안 된다.
   PCA 전처리로 성능을 유지하면서 속도를 높일 수 있다.
4. 텍스트 표현: SBERT는 의미 유사도를 반영하고, 특히 적은 데이터에서 우위가 크다.
   TF-IDF → SBERT 순서로 시도하는 것이 실무적 접근이다.
5. AI 도구로 코드를 생성하되, 결과를 반드시 검증하고 해석하는 습관이 중요하다.
```
