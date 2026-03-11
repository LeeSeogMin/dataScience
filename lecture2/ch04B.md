# 4장 B: 군집 분석 — 모범 답안과 해설

> 이 문서는 실습 제출 후 공개한다. 제출 전에는 열람하지 않는다.

---

## 실습 1 해설: HDBSCAN 군집 분석

### 제공 코드 실행 결과 해설

반달 모양 데이터(500포인트, 2개 군집 + 20% 노이즈)에 K-Means와 HDBSCAN을 적용한 결과:

| 항목 | K-Means | HDBSCAN | 이유 |
| ---- | ------- | ------- | ---- |
| 군집 수 | 2 (사전 지정) | 2 (자동 탐지) | HDBSCAN이 밀도 기반으로 올바르게 2개 군집 발견 |
| 노이즈 탐지 | 0 | 약 100개 | K-Means는 모든 점을 강제 할당, HDBSCAN은 노이즈 자동 분리 |
| ARI | 낮음 (0.3~0.5) | 높음 (0.7~0.9) | K-Means는 구형 경계로 반달 모양 분리 실패 |
| 실루엣 | 낮음 | 높음 | HDBSCAN은 밀도 구조에 맞게 군집화 |

핵심 코드 구조:

```python
# K-Means: 구형 경계, 노이즈도 강제 할당
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X)

# HDBSCAN: 밀도 기반, 노이즈 자동 분리 (-1)
clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5)
labels_hdbscan = clusterer.fit_predict(X)
```

구형 군집에서는 두 알고리즘 모두 ARI가 높다. 이는 K-Means의 구형 가정이 실제 데이터 형태와 일치하기 때문이다. HDBSCAN 파라미터 분석에서 min_cluster_size를 키우면 군집 수가 줄고 노이즈가 증가한다. 이는 "최소 이만큼의 포인트가 있어야 군집"이라는 기준이 엄격해지기 때문이다.

### 프롬프트 1 모범 구현: min_cluster_size 변화 실험

```python
import numpy as np
import hdbscan
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score

np.random.seed(42)

# 반달 데이터 + 노이즈 생성
X_moons, y_moons = make_moons(n_samples=400, noise=0.08, random_state=42)
X_noise = np.random.uniform(-1.5, 2.5, size=(100, 2))
y_noise = np.array([-1] * 100)
X = np.vstack([X_moons, X_noise])
y_true = np.concatenate([y_moons, y_noise])

min_cluster_sizes = [5, 10, 20, 40, 80]

print(f"{'min_cluster_size':<20} {'군집 수':>10} {'노이즈 비율':>12} {'ARI':>10}")
print("-" * 52)

for mcs in min_cluster_sizes:
    clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, min_samples=5)
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = (labels == -1).sum() / len(labels)

    # 노이즈 제외하고 ARI 계산
    mask = (y_true != -1) & (labels != -1)
    if mask.sum() > 0 and len(set(labels[mask])) > 1:
        ari = adjusted_rand_score(y_true[mask], labels[mask])
    else:
        ari = 0.0

    print(f"{mcs:<20} {n_clusters:>10} {noise_ratio:>12.1%} {ari:>10.4f}")
```

기대 결과 해석:
- min_cluster_size=5: 작은 군집도 허용 → 군집 수가 많아질 수 있고, 노이즈 비율이 낮음
- min_cluster_size=20: 적절한 설정. 2개 군집 + 노이즈 탐지. ARI 가장 높음
- min_cluster_size=80: 너무 엄격 → 군집 수가 줄거나 1개가 될 수 있음, 노이즈 비율 증가
- 핵심: min_cluster_size는 "최소 몇 개 이상이어야 진짜 군집인가"를 결정하는 파라미터. 데이터 크기와 예상 군집 크기를 고려해 설정한다

### 프롬프트 2 모범 구현: K-Means 엘보우 분석

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

np.random.seed(42)
X_moons, y_moons = make_moons(n_samples=400, noise=0.08, random_state=42)
X_noise = np.random.uniform(-1.5, 2.5, size=(100, 2))
X = np.vstack([X_moons, X_noise])

K_range = range(2, 9)
wcss = []
sil_scores = []

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(K_range, wcss, 'bo-')
ax1.set_xlabel('K')
ax1.set_ylabel('WCSS (Inertia)')
ax1.set_title('Elbow Method')
ax1.grid(True, alpha=0.3)

ax2.plot(K_range, sil_scores, 'ro-')
ax2.set_xlabel('K')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Analysis')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("kmeans_elbow_analysis.png", dpi=150)
plt.show()

print(f"최적 K (실루엣 기준): {K_range[np.argmax(sil_scores)]}")
print(f"실제 군집 수: 2")
```

기대 결과:
- 엘보우가 K=2에서 나타나면 실제 군집 수와 일치하지만, 노이즈가 포함된 데이터에서는 엘보우가 불명확할 수 있다
- 실루엣 점수가 K=2에서 최고가 아닐 수 있다. 반달 모양은 K-Means에 부적합하므로 모든 K에서 실루엣이 낮다
- 이것이 "K 결정 방법의 한계": 데이터가 K-Means 가정(구형)에 맞지 않으면 최적 K를 찾아도 좋은 결과를 얻기 어렵다

---

## 실습 2 해설: 임베딩 기반 고객 리뷰 군집화

### 제공 코드 실행 결과 해설

40개 상품 리뷰(배송/품질/가격/서비스/만족도 5개 주제)를 Sentence-BERT + UMAP + HDBSCAN으로 군집화한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 발견 군집 수 | 2~3개 | 40개 리뷰로는 5개 주제를 모두 분리하기 어려움. 의미적으로 유사한 주제가 합쳐짐 |
| 노이즈 포인트 | 0~5개 | 리뷰가 적어 HDBSCAN이 일부를 군집에 포함시키지 못할 수 있음 |
| ARI | 0.05~0.20 | 5개 주제 vs 2~3개 군집이므로 완벽한 일치는 기대하기 어려움 |
| TF-IDF vs 임베딩 | 임베딩이 높음 | 임베딩이 "빨랐어요"/"빠른 배달"을 같은 의미로 인식 |

핵심 코드 구조:

```python
# 파이프라인: 텍스트 → 임베딩 → 차원 축소 → 군집화
embeddings = get_embeddings(texts, method='sbert')          # 384차원
embeddings_reduced = reduce_dimensions(embeddings, n_components=10)  # 10차원
labels = cluster_embeddings(embeddings_reduced)              # HDBSCAN
```

HDBSCAN이 5개가 아닌 2~3개 군집을 발견하는 것은 "실패"가 아니다. 의미적으로 "배송이 빨라서 좋았어요"와 "고객센터가 친절해요"는 서비스 경험이라는 공통 의미를 공유한다. HDBSCAN은 임베딩 공간에서 실제로 분리되는 군집만 발견하므로, 이 결과가 데이터의 의미적 구조를 더 정확히 반영한다.

### 프롬프트 3 모범 구현: UMAP n_components 실험

```python
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

# 리뷰 데이터 (4-4-embedding-clustering.py에서 가져옴)
reviews = [
    "배송이 정말 빨라서 좋았어요. 다음날 바로 도착했습니다.",
    "배송 속도가 엄청 빠르네요. 주문한 지 하루 만에 왔어요.",
    "택배가 너무 늦게 왔어요. 일주일이나 걸렸습니다.",
    "배송이 예상보다 빨라서 만족합니다.",
    "배송 추적이 안 돼서 불안했어요. 택배사 문제 같아요.",
    "당일 배송으로 받아서 급한 거 해결했어요.",
    "배송 중 파손되어 왔어요. 포장을 더 신경 써주세요.",
    "새벽 배송 너무 편리해요. 출근 전에 받을 수 있어서 좋아요.",
    "제품 품질이 가격 대비 정말 좋아요. 추천합니다.",
    "품질이 기대 이상이에요. 마감도 깔끔하고 튼튼해요.",
    "품질이 별로네요. 사진이랑 많이 달라요.",
    "소재가 고급스럽고 마감이 꼼꼼해요.",
    "사진보다 색상이 많이 다르네요. 실망했어요.",
    "내구성이 좋아서 오래 쓸 수 있을 것 같아요.",
    "품질 좋고 디자인도 예뻐요. 재구매 의향 있습니다.",
    "처음엔 좋았는데 한 달 만에 망가졌어요.",
    "가격 대비 성능이 좋아요. 가성비 최고입니다.",
    "이 가격에 이 품질이면 완전 이득이에요.",
    "좀 비싼 것 같아요. 할인할 때 사는 게 좋겠어요.",
    "가성비 최고! 다른 데보다 훨씬 저렴해요.",
    "가격은 비싸지만 그만한 가치가 있어요.",
    "세일할 때 샀는데 정가에는 안 살 것 같아요.",
    "무료 배송이라 더 저렴하게 느껴져요.",
    "비슷한 제품 중 가장 합리적인 가격이에요.",
    "고객센터 응대가 정말 친절했어요. 문제 해결도 빨랐고요.",
    "교환 과정이 너무 복잡해요. 개선이 필요해요.",
    "AS 신청했는데 빠르게 처리해주셔서 감사해요.",
    "문의 답변이 너무 늦어요. 3일이나 걸렸어요.",
    "환불 요청했는데 바로 처리해주셨어요.",
    "상담원분이 친절하게 설명해주셔서 좋았어요.",
    "반품 절차가 간단해서 좋았어요.",
    "고객센터 연결이 너무 어려워요. 전화를 안 받아요.",
    "써보니까 진짜 편해요. 매일 쓰고 있어요.",
    "기대했던 것보다 훨씬 좋아요. 완전 만족합니다.",
    "생각보다 별로예요. 기대가 컸나 봐요.",
    "가족들도 다 좋아해요. 추가로 더 살 예정이에요.",
    "사용법이 간단해서 누구나 쓸 수 있어요.",
    "첫 사용감이 좋아요. 앞으로도 계속 쓸 것 같아요.",
    "선물용으로 샀는데 받으신 분이 너무 좋아하세요.",
    "재구매했어요. 그만큼 만족스러워요.",
]

true_topics = ['배송']*8 + ['품질']*8 + ['가격']*8 + ['서비스']*8 + ['만족도']*8
topic_map = {t: i for i, t in enumerate(set(true_topics))}
y_true = np.array([topic_map[t] for t in true_topics])

# 임베딩 생성
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(reviews, normalize_embeddings=True)

n_components_list = [2, 5, 10, 20, 50]

print(f"{'n_components':<15} {'군집 수':>10} {'노이즈 수':>12} {'ARI':>10}")
print("-" * 47)

for nc in n_components_list:
    reducer = umap.UMAP(n_components=nc, n_neighbors=15, min_dist=0.1,
                        metric='cosine', random_state=42)
    reduced = reducer.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
    labels = clusterer.fit_predict(reduced)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()

    mask = labels != -1
    if mask.sum() > 0 and len(set(labels[mask])) > 1:
        ari = adjusted_rand_score(y_true[mask], labels[mask])
    else:
        ari = 0.0

    print(f"{nc:<15} {n_clusters:>10} {n_noise:>12} {ari:>10.4f}")
```

기대 결과 해석:
- n_components=2: 너무 많은 정보를 잃어 군집 구조가 무너질 수 있음. 시각화에는 좋지만 군집화에는 부족
- n_components=10: 적절한 균형. 대부분의 군집 구조가 보존됨
- n_components=50: 40개 데이터에 50차원은 차원이 오히려 너무 높음. 차원의 저주로 거리 측정이 무의미해질 수 있음
- 데이터 수보다 차원이 높으면 안 된다는 실용적 규칙을 확인할 수 있다

### 프롬프트 4 모범 구현: 코사인 유사도 분석

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

# (리뷰 데이터와 임베딩 생성 코드는 프롬프트 3과 동일)
# ... (생략)

# 군집화 수행
reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1,
                    metric='cosine', random_state=42)
reduced = reducer.fit_transform(embeddings)
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3)
labels = clusterer.fit_predict(reduced)

# 원본 임베딩에서 코사인 유사도 계산
sim_matrix = cosine_similarity(embeddings)

# 같은 군집 내 평균 유사도
intra_sims = []
inter_sims = []

for i in range(len(labels)):
    for j in range(i + 1, len(labels)):
        if labels[i] == -1 or labels[j] == -1:
            continue
        if labels[i] == labels[j]:
            intra_sims.append(sim_matrix[i, j])
        else:
            inter_sims.append(sim_matrix[i, j])

print(f"군집 내 평균 코사인 유사도: {np.mean(intra_sims):.4f}")
print(f"군집 간 평균 코사인 유사도: {np.mean(inter_sims):.4f}")
print(f"차이: {np.mean(intra_sims) - np.mean(inter_sims):.4f}")

if np.mean(intra_sims) > np.mean(inter_sims):
    print("→ 군집 내 유사도가 군집 간 유사도보다 높음 (좋은 군집화)")
else:
    print("→ 군집 내 유사도가 군집 간 유사도보다 낮음 (군집화 개선 필요)")
```

기대 결과:
- 군집 내 평균 유사도 > 군집 간 평균 유사도
- 차이가 클수록 군집화 품질이 높음
- 40개 리뷰로는 차이가 크지 않을 수 있지만, 데이터가 많아지면 더 명확해진다

### 프롬프트 5 모범 구현: TF-IDF vs 임베딩 시각화 비교

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sentence_transformers import SentenceTransformer
import umap
import hdbscan

# (리뷰 데이터 로드 코드 동일)
# ... (생략)

# TF-IDF + K-Means
tfidf = TfidfVectorizer(max_features=500)
X_tfidf = tfidf.fit_transform(reviews).toarray()
labels_tfidf = KMeans(n_clusters=5, random_state=42, n_init=10).fit_predict(X_tfidf)
ari_tfidf = adjusted_rand_score(y_true, labels_tfidf)

# 임베딩 + HDBSCAN
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(reviews, normalize_embeddings=True)
reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1,
                    metric='cosine', random_state=42)
reduced = reducer.fit_transform(embeddings)
labels_emb = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=3).fit_predict(reduced)
mask = labels_emb != -1
ari_emb = adjusted_rand_score(y_true[mask], labels_emb[mask]) if mask.sum() > 0 else 0

# 2D 시각화용 축소
reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                       metric='cosine', random_state=42)
emb_2d = reducer_2d.fit_transform(embeddings)

pca_2d = PCA(n_components=2, random_state=42)
tfidf_2d = pca_2d.fit_transform(X_tfidf)

topic_colors = {'배송': 'red', '품질': 'blue', '가격': 'green',
                '서비스': 'purple', '만족도': 'orange'}
colors = [topic_colors[t] for t in true_topics]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(tfidf_2d[:, 0], tfidf_2d[:, 1], c=colors, s=80, alpha=0.7, edgecolors='white')
ax1.set_title(f'TF-IDF + K-Means (ARI={ari_tfidf:.3f})', fontweight='bold')
ax1.set_xlabel('PCA 1')
ax1.set_ylabel('PCA 2')

ax2.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=80, alpha=0.7, edgecolors='white')
ax2.set_title(f'Embedding + HDBSCAN (ARI={ari_emb:.3f})', fontweight='bold')
ax2.set_xlabel('UMAP 1')
ax2.set_ylabel('UMAP 2')

# 범례
for topic, color in topic_colors.items():
    ax2.scatter([], [], c=color, label=topic, s=80)
ax2.legend(loc='best', fontsize=8)

plt.tight_layout()
plt.savefig("tfidf_vs_embedding_comparison.png", dpi=150)
plt.show()
```

기대 결과:
- TF-IDF(왼쪽): 주제별 색상이 섞여 있음. 같은 주제라도 다른 단어를 사용하면 멀리 떨어짐
- 임베딩(오른쪽): 같은 주제의 점들이 더 가깝게 모임. 의미적 유사성을 반영
- ARI가 임베딩 쪽이 높을 것으로 예상

---

## 실습 3 해설: 딥 클러스터링 알고리즘 비교

### 제공 코드 실행 결과 해설

두 가지 텍스트 데이터셋(실험 A: 쇼핑 리뷰 주제, 실험 B: 다른 도메인)에 K-Means, DEC, IDEC, VaDE를 적용한 결과:

| 알고리즘 | 실험 A ARI | 실험 B ARI | 핵심 |
| -------- | ---------- | ---------- | ---- |
| KMeans | 0.01~0.02 | 0.13~0.14 | 베이스라인. 임베딩에 직접 적용 |
| DEC | 0.04~0.05 | 0.16~0.17 | KMeans 대비 개선되지만 절대값은 낮음 |
| IDEC | 0.03~0.05 | 0.09~0.10 | 재구성 손실 추가로 안정적이지만 DEC보다 낮을 수 있음 |
| VaDE | ~0.00 | ~0.00 | 모든 샘플을 단일 군집에 할당. 학습 실패 |

핵심 코드 구조:

```python
# 1. 오토인코더 사전학습: 데이터 압축 능력 학습
ae = Autoencoder(input_dim, hidden_dims=[256, 128, 64], latent_dim=32)
ae = pretrain_autoencoder(ae, dataloader, epochs=30)

# 2. 잠재 공간에서 K-Means 초기화
kmeans_init = KMeans(n_clusters=5).fit(z_init)
cluster_layer.cluster_centers.data = torch.FloatTensor(kmeans_init.cluster_centers_)

# 3. DEC 학습: KL 발산으로 군집 특화 표현 학습
q = cluster_layer(z)              # Student's t-분포 소프트 할당
p = target_distribution(q.detach()) # 더 확신 높은 타깃 분포
loss = KL(P || Q)                   # Q가 P를 따르도록 학습
```

실험 A의 ARI가 전반적으로 낮은 이유: Sentence-BERT 임베딩이 "쇼핑 경험"이라는 공통 도메인에 집중하여, "배송이 빨라서 좋아요"와 "품질이 좋아요"를 모두 "긍정적 쇼핑 경험"으로 가깝게 배치한다. 이것은 모델의 "실패"가 아니라 학습 목적(의미 유사도)과 사용 목적(세부 주제 구분)의 불일치다.

VaDE가 실패하는 이유: VAE의 ELBO 최적화와 GMM 파라미터 초기화가 복잡하여, 하이퍼파라미터 튜닝 없이는 posterior collapse(잠재 공간이 사전 분포에 수렴)가 발생한다. 이론적으로는 우수하지만 실무 적용이 까다롭다.

### 프롬프트 6 모범 구현: t-SNE 임베딩 공간 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# 데이터 생성 (4-5 코드에서 함수 가져오기)
# generate_review_topics()와 generate_different_domains() 사용
np.random.seed(42)

# 실험 A 데이터 (간소화)
from importlib.machinery import SourceFileLoader
import sys, os

# 직접 데이터 생성 (실제로는 4-5 코드의 함수 재사용)
# ... (generate_review_topics, generate_different_domains 함수)

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# 실험 A 임베딩 (3000개 샘플링)
sample_A = np.random.choice(len(reviews_A), 3000, replace=False)
emb_A = model.encode([reviews_A[i] for i in sample_A], normalize_embeddings=True)
y_A = y_true_A[sample_A]

# 실험 B 임베딩
sample_B = np.random.choice(len(reviews_B), 3000, replace=False)
emb_B = model.encode([reviews_B[i] for i in sample_B], normalize_embeddings=True)
y_B = y_true_B[sample_B]

# t-SNE 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

tsne_A = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb_A)
tsne_B = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(emb_B)

colors = plt.cm.tab10(np.linspace(0, 1, 5))

for i in range(5):
    mask = y_A == i
    ax1.scatter(tsne_A[mask, 0], tsne_A[mask, 1], c=[colors[i]], s=5, alpha=0.5,
                label=labels_A[i])
ax1.set_title('실험 A: 쇼핑 리뷰 주제\n(임베딩 공간에서 주제 분리 불명확)', fontweight='bold')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

for i in range(5):
    mask = y_B == i
    ax2.scatter(tsne_B[mask, 0], tsne_B[mask, 1], c=[colors[i]], s=5, alpha=0.5,
                label=labels_B[i])
ax2.set_title('실험 B: 다른 도메인\n(임베딩 공간에서 도메인 분리 명확)', fontweight='bold')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("tsne_embedding_comparison.png", dpi=150)
plt.show()
```

기대 결과:
- 실험 A: 5개 주제의 색상이 섞여 있음. 특히 배송/서비스/만족도가 겹침
- 실험 B: 5개 도메인이 명확하게 분리된 군집을 형성
- 이 시각화로 "군집화 전에 임베딩 공간의 분리 상태를 확인해야 한다"는 실무 지침을 체감할 수 있다

### 프롬프트 7 모범 구현: 잠재 차원 크기 실험

```python
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sentence_transformers import SentenceTransformer

# 실험 B 데이터 생성 (간소화: 5000개)
np.random.seed(42)
# reviews_B, y_true_B, labels_B = generate_different_domains(n_samples=5000)
# embeddings = get_embeddings(reviews_B)
# ... (데이터 생성 코드 생략)

# Autoencoder, ClusteringLayer, target_distribution 정의는 4-5 코드와 동일
# ... (모델 정의 생략)

latent_dims = [8, 16, 32, 64]
input_dim = embeddings.shape[1]

print(f"{'latent_dim':<14} {'ARI':>10}")
print("-" * 24)

for ld in latent_dims:
    torch.manual_seed(42)

    # 오토인코더 사전학습
    ae = Autoencoder(input_dim, hidden_dims=[256, 128, 64], latent_dim=ld)
    X_tensor = torch.FloatTensor(embeddings)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

    ae = pretrain_autoencoder(ae, dataloader, epochs=30)

    # K-Means 초기화
    ae.eval()
    with torch.no_grad():
        _, z_init = ae(X_tensor)
        z_init = z_init.numpy()
    kmeans_init = KMeans(n_clusters=5, random_state=42, n_init=10).fit(z_init)

    # DEC 학습
    cluster_layer = ClusteringLayer(5, ld)
    cluster_layer.cluster_centers.data = torch.FloatTensor(kmeans_init.cluster_centers_)
    ae, cluster_layer = train_dec(ae, cluster_layer, dataloader, epochs=50)

    # 평가
    ae.eval()
    cluster_layer.eval()
    with torch.no_grad():
        _, z = ae(X_tensor)
        q = cluster_layer(z)
        labels = q.argmax(dim=1).numpy()

    ari = adjusted_rand_score(y_true_B, labels)
    print(f"{ld:<14} {ari:>10.4f}")
```

기대 결과:
- latent_dim=8: 384차원을 8차원으로 압축하면 정보 손실이 큼. ARI 낮음
- latent_dim=32: 적절한 균형. 대부분 가장 높은 ARI
- latent_dim=64: 차원이 높아지면서 군집 경계가 희석될 수 있음. ARI가 32와 비슷하거나 약간 낮음
- 핵심: 잠재 차원은 "적당히 좁은 병목"이어야 핵심 구조만 남기면서 노이즈를 제거한다

### 프롬프트 8 모범 구현: K-Means vs DEC 혼동행렬

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns

# (임베딩 생성 및 DEC 학습 코드는 위와 동일)
# labels_kmeans: K-Means 결과
# labels_dec: DEC 결과

# 두 알고리즘 간 혼동행렬
cm = confusion_matrix(labels_kmeans, labels_dec)

# 같은 군집에 할당된 비율
# 최적 매칭을 위해 Hungarian algorithm 사용
from scipy.optimize import linear_sum_assignment
row_ind, col_ind = linear_sum_assignment(-cm)
agreement = cm[row_ind, col_ind].sum() / len(labels_kmeans)

print(f"K-Means vs DEC 일치율: {agreement:.1%}")
print(f"K-Means ARI: {adjusted_rand_score(y_true_B, labels_kmeans):.4f}")
print(f"DEC ARI: {adjusted_rand_score(y_true_B, labels_dec):.4f}")

# 혼동행렬 시각화
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'DEC {i}' for i in range(5)],
            yticklabels=[f'KM {i}' for i in range(5)])
plt.xlabel('DEC 군집')
plt.ylabel('K-Means 군집')
plt.title(f'K-Means vs DEC 군집 할당 비교 (일치율: {agreement:.1%})')
plt.tight_layout()
plt.savefig("kmeans_vs_dec_confusion.png", dpi=150)
plt.show()
```

기대 결과:
- 일치율 60~80%: 두 알고리즘이 대체로 비슷한 군집을 형성하지만, DEC가 일부 경계 샘플을 다르게 할당
- DEC의 ARI가 더 높다면: DEC가 잠재 공간을 재학습하여 군집 경계를 더 잘 잡은 것
- 대각선 값이 높은 혼동행렬: 두 알고리즘의 군집이 잘 대응됨
- Hungarian algorithm으로 최적 매칭을 찾는 이유: 군집 레이블 번호가 다를 수 있으므로 (K-Means의 군집 0이 DEC의 군집 3에 대응)

---

## 4장 전체 핵심 정리

```text
1. 군집 분석: K-Means는 구형 군집에 빠르고 효과적이지만, 비구형 군집과 노이즈에는
   HDBSCAN이 우수하다. min_cluster_size가 군집 결과에 큰 영향을 미친다.
2. 임베딩 기반 군집화: TF-IDF보다 의미적 유사성을 잘 포착한다. UMAP 차원은
   데이터 수보다 낮게, 2보다는 높게 설정한다. 10~50이 일반적.
3. 딥 클러스터링: DEC/IDEC는 K-Means 대비 개선되지만, 임베딩 자체가 군집 구조를
   포착하지 못하면 근본적 한계가 있다. VaDE는 학습 불안정으로 실무 적용이 어렵다.
4. 핵심 판단: "임베딩이 포착하는 유사성"과 "군집화 목적의 유사성"이 일치하는가?
   불일치하면 도메인 특화 임베딩이나 fine-tuned 모델을 먼저 고려한다.
5. 실무 순서: K-Means 베이스라인 → t-SNE 시각화로 분리 확인 → 필요시 HDBSCAN
   또는 딥 클러스터링 → 결과의 비즈니스 해석 가능성 검증.
6. AI 도구로 코드를 생성하되, 결과를 반드시 검증하고 해석하는 습관이 중요하다.
```
