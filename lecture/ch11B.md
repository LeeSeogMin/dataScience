# 11장 B: 그래프 분석과 학습 — 모범 답안과 해설

> 이 문서는 실습 제출 후 공개한다. 제출 전에는 열람하지 않는다.

---

## 실습 1 해설: Karate Club 중심성 분석

### 제공 코드 실행 결과 해설

Zachary's Karate Club(34노드, 78엣지, 밀도 0.1390)에 4가지 중심성을 계산한 결과:

| 중심성 지표 | 1위 노드 | 점수 | 2위 노드 | 점수 |
| ----------- | -------- | ---- | -------- | ---- |
| 연결 중심성 | 노드 33 | 0.515 | 노드 0 | 0.485 |
| 매개 중심성 | 노드 0 | 0.438 | 노드 33 | 0.304 |
| 근접 중심성 | 노드 0 | 0.569 | 노드 2 | 0.559 |
| PageRank | 노드 33 | 0.097 | 노드 0 | 0.089 |

핵심 해석:
- 노드 33(Officer)은 직접 연결 수(17명)와 중요한 노드들로부터의 연결(PageRank)에서 우위
- 노드 0(Mr. Hi)은 정보 흐름 중개(매개 중심성)와 네트워크 중심 위치(근접 중심성)에서 우위
- 두 리더는 서로 다른 방식으로 영향력을 행사: Mr. Hi는 "소통의 관문", Officer는 "사회적 인기"
- 노드 32는 모든 지표에서 3위 — 두 리더 사이의 "숨겨진 브리지" 역할

### 프롬프트 1 모범 구현: 노드 제거 시 네트워크 영향 분석

```python
import networkx as nx

G = nx.karate_club_graph()

# 매개 중심성 상위 3개 노드 추출
betweenness = nx.betweenness_centrality(G)
top_3 = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]

print(f"{'제거 노드':<16} {'연결 요소 수':<14} {'평균 최단 경로 길이'}")
print("-" * 50)

# 원본 그래프
avg_path = nx.average_shortest_path_length(G)
print(f"{'원본 (없음)':<16} {1:<14} {avg_path:<.4f}")

for node, score in top_3:
    G_removed = G.copy()
    G_removed.remove_node(node)

    n_components = nx.number_connected_components(G_removed)

    # 가장 큰 연결 요소의 평균 최단 경로
    largest_cc = max(nx.connected_components(G_removed), key=len)
    G_largest = G_removed.subgraph(largest_cc)
    avg_path_removed = nx.average_shortest_path_length(G_largest)

    print(f"노드 {node} (B={score:.3f})  {n_components:<14} {avg_path_removed:<.4f}")
```

기대 결과 해석:
- 매개 중심성 1위(노드 0) 제거 시: 연결 요소가 여러 개로 분리되거나 평균 경로가 크게 증가
- 이는 조직에서 핵심 중개자의 이탈이 정보 흐름에 미치는 영향을 보여줌
- 실무 시사점: 매개 중심성이 높은 직원에 대해 지식 이전과 백업 인력 배치가 필요

---

## 실습 2 해설: Karate Club 커뮤니티 탐지

### 제공 코드 실행 결과 해설

Louvain 알고리즘 적용 결과:

| 항목 | 값 | 해석 |
| ---- | -- | ---- |
| 커뮤니티 수 | 4개 | 두 주요 그룹 + 두 서브그룹 |
| 모듈성 (Q) | 0.444 | 0.3 이상으로 유의미한 구조 |
| 실제 분열 정확도 | 82.4% | 알고리즘이 분열을 잘 예측 |

커뮤니티별 통계:
- 커뮤니티 0 (Mr. Hi 파): 11노드, 밀도 0.418 — 강사를 중심으로 한 핵심 그룹
- 커뮤니티 3 (Officer 파): 14노드, 밀도 0.286 — 회장을 중심으로 한 큰 그룹 (밀도가 낮은 것은 규모가 크기 때문)
- 커뮤니티 1, 2: 각각 5, 4노드의 작은 서브그룹. 메인 그룹과 연결되지만 독립적인 하위 구조

### 프롬프트 2 모범 구현: 해상도 파라미터 시각화

```python
import networkx as nx
import matplotlib.pyplot as plt
import community as community_louvain

G = nx.karate_club_graph()
pos = nx.spring_layout(G, seed=42)

resolutions = [0.5, 1.0, 1.5, 2.0]
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

gray_colors = ['white', 'lightgray', 'darkgray', 'black', 'silver', 'dimgray', 'gainsboro']

for ax, res in zip(axes.flatten(), resolutions):
    partition = community_louvain.best_partition(G, resolution=res, random_state=42)
    modularity = community_louvain.modularity(partition, G)
    n_comm = len(set(partition.values()))

    node_colors = [gray_colors[partition[node] % len(gray_colors)] for node in G.nodes()]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=400,
                           node_color=node_colors, edgecolors='black', linewidths=1.5)
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='gray', alpha=0.5)

    for node in G.nodes():
        comm = partition[node]
        font_color = 'white' if comm == 3 else 'black'
        nx.draw_networkx_labels(G, pos, labels={node: node}, ax=ax,
                                font_size=8, font_color=font_color)

    ax.set_title(f"Resolution={res}\n{n_comm} communities, Q={modularity:.3f}",
                 fontsize=11, fontweight="bold")
    ax.axis("off")

plt.tight_layout()
plt.savefig("resolution_comparison.png", dpi=150)
plt.show()
```

기대 결과:
- resolution=0.5: 큰 커뮤니티가 주로 탐지됨 (4~5개)
- resolution=1.0: 기본값, 4개 커뮤니티, 모듈성 최대 근처
- resolution=1.5: 5개 커뮤니티로 분리 시작
- resolution=2.0: 7개의 작은 커뮤니티, 모듈성 감소
- 실무에서는 분석 목적(고객 세그먼트 크기 등)에 맞는 해상도를 선택

---

## 실습 3 해설: 링크 예측 성능 평가

### 제공 코드 실행 결과 해설

엣지를 80:20으로 분할하고 4가지 지표로 제거된 엣지를 예측한 결과:

| 방법 | AUC-ROC | 평균 정밀도 | 해석 |
| ---- | ------- | ----------- | ---- |
| 공통 이웃 | 0.684 | 0.613 | 기본적이지만 안정적 |
| Jaccard | 0.624 | 0.564 | 정규화로 허브 편향 보정, 소규모에서 불리 |
| Adamic-Adar | 0.696 | 0.706 | 평균 정밀도 최고 — 희귀 이웃 가중치의 효과 |
| Preferential Attachment | 0.707 | 0.630 | AUC 최고 — 허브 노드끼리의 연결 경향 반영 |

핵심: AUC 0.5가 무작위, 1.0이 완벽 예측이므로 모든 지표가 0.6 이상이면 네트워크 구조가 예측에 유용하다는 증거다. Karate Club에서 Preferential Attachment가 우수한 것은 소규모 밀집 네트워크에서 허브 연결 경향이 강하기 때문이다.

### 프롬프트 3 모범 구현: Adamic-Adar 상위 노드 쌍 분석

```python
import random
import networkx as nx

random.seed(42)

G = nx.karate_club_graph()

# 학습용 그래프 생성 (20% 엣지 제거)
edges = list(G.edges())
n_test = int(len(edges) * 0.2)
G_train = G.copy()

random.shuffle(edges)
removed_edges = set()
for edge in edges:
    if len(removed_edges) >= n_test:
        break
    G_train.remove_edge(*edge)
    if nx.is_connected(G_train):
        removed_edges.add(edge)
        removed_edges.add((edge[1], edge[0]))
    else:
        G_train.add_edge(*edge)

# 연결되지 않은 모든 노드 쌍에 대해 Adamic-Adar 점수 계산
non_edges = list(nx.non_edges(G_train))
preds = nx.adamic_adar_index(G_train, non_edges)
scores = [(u, v, score) for u, v, score in preds]
scores.sort(key=lambda x: x[2], reverse=True)

# 상위 10개 출력
print(f"{'순위':<6} {'노드 쌍':<16} {'AA 점수':<12} {'원본에서 연결'}")
print("-" * 50)

hit_count = 0
for i, (u, v, score) in enumerate(scores[:10], 1):
    in_original = G.has_edge(u, v)
    if in_original:
        hit_count += 1
    marker = "O" if in_original else "X"
    print(f"{i:<6} ({u:>2}, {v:>2})       {score:<12.4f} {marker}")

print(f"\n상위 10개 중 실제 연결된 쌍: {hit_count}/10 ({hit_count/10:.0%})")
```

기대 결과:
- 상위 10개 예측 중 실제로 연결된(제거되었던) 쌍이 여러 개 포함됨
- 이 비율이 링크 예측의 Precision@10에 해당
- Adamic-Adar가 희귀 공통 이웃에 가중치를 주어 정확한 예측을 가능하게 함

### 프롬프트 4 모범 구현: ROC 곡선 시각화

```python
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

random.seed(42)
np.random.seed(42)

G = nx.karate_club_graph()

# 학습/테스트 분할
edges = list(G.edges())
n_test = int(len(edges) * 0.2)
G_train = G.copy()
test_edges = []

random.shuffle(edges)
for edge in edges:
    if len(test_edges) >= n_test:
        break
    G_train.remove_edge(*edge)
    if nx.is_connected(G_train):
        test_edges.append(edge)
    else:
        G_train.add_edge(*edge)

# 네거티브 샘플 생성
nodes = list(G_train.nodes())
existing = set(G_train.edges()) | set((v, u) for u, v in G_train.edges())
neg_edges = []
while len(neg_edges) < len(test_edges):
    u, v = random.sample(nodes, 2)
    if (u, v) not in existing:
        neg_edges.append((u, v))
        existing.add((u, v))
        existing.add((v, u))

all_edges = test_edges + neg_edges
labels = [1] * len(test_edges) + [0] * len(neg_edges)

methods = {
    "Common Neighbors": nx.common_neighbor_centrality,
    "Jaccard": nx.jaccard_coefficient,
    "Adamic-Adar": nx.adamic_adar_index,
    "Preferential Attachment": nx.preferential_attachment
}

fig, ax = plt.subplots(figsize=(8, 7))
linestyles = ['-', '--', '-.', ':']

for (name, func), ls in zip(methods.items(), linestyles):
    scores = [s for _, _, s in func(G_train, all_edges)]
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    ax.plot(fpr, tpr, linestyle=ls, linewidth=2, color='black',
            label=f"{name} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.500)')
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.set_title("Link Prediction ROC Curves", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig("link_prediction_roc.png", dpi=150)
plt.show()
```

기대 결과: ROC 곡선에서 Preferential Attachment과 Adamic-Adar가 대각선(무작위)에서 가장 멀리 떨어져 있어 최고 성능을 시각적으로 확인할 수 있다.

---

## 실습 4 해설: Node2Vec 임베딩과 유사 노드 검색

### 제공 코드 실행 결과 해설

Node2Vec(dimensions=64, walk_length=30, num_walks=200, p=1, q=1) 결과:

| 노드 0(Mr. Hi)과 유사 | 유사도 | 노드 33(Officer)과 유사 | 유사도 |
| ---------------------- | ------ | ----------------------- | ------ |
| 노드 17 | 0.789 | 노드 32 | 0.868 |
| 노드 11 | 0.728 | 노드 22 | 0.851 |
| 노드 10 | 0.716 | 노드 20 | 0.845 |
| 노드 1 | 0.712 | 노드 14 | 0.839 |
| 노드 4 | 0.705 | 노드 15 | 0.797 |

핵심: 유사 노드들이 실제로 각 리더의 진영에 속한 멤버들이다. 임베딩이 커뮤니티 구조를 잘 포착한다는 증거.

p, q 파라미터 비교: 작은 네트워크에서는 설정에 따른 ARI 차이가 크지 않다 (모두 0.6 전후). 대규모 희소 네트워크에서는 BFS-like가 지역 구조, DFS-like가 전역 커뮤니티 포착에 유리하다.

### 프롬프트 5 모범 구현: 임베딩 차원 변경 실험

```python
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import community as community_louvain

G = nx.karate_club_graph()

# 실제 커뮤니티 레이블
true_partition = community_louvain.best_partition(G, random_state=42)
true_labels = [true_partition[n] for n in G.nodes()]

dimensions_list = [8, 16, 32, 64, 128]

print(f"{'임베딩 차원':<14} {'ARI':<10}")
print("-" * 24)

for dim in dimensions_list:
    n2v = Node2Vec(G, dimensions=dim, walk_length=20, num_walks=100,
                   p=1, q=1, workers=1, seed=42, quiet=True)
    model = n2v.fit(window=10, min_count=1, batch_words=4)

    vectors = np.array([model.wv[str(n)] for n in G.nodes()])
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(vectors)

    ari = adjusted_rand_score(true_labels, pred_labels)
    print(f"{dim:<14} {ari:<10.4f}")
```

기대 결과:
- 차원이 너무 작으면(8) 구조 정보를 충분히 담지 못해 ARI가 낮을 수 있음
- 차원 32~64에서 안정적인 성능
- 차원 128은 작은 네트워크에서는 과도 — 노이즈가 증가할 수 있음
- 적절한 차원은 네트워크 크기와 복잡도에 비례하여 선택

---

## 실습 5 해설: MovieLens 협업 필터링

### 제공 코드 실행 결과 해설

MovieLens 100K(943 사용자, 1,682 아이템, 10만 평점, 희소성 93.7%):

| 모델 | RMSE | MAE | 해석 |
| ---- | ---- | --- | ---- |
| SVD | 0.932 ± 0.002 | 0.734 ± 0.002 | 잠재 공간 압축으로 노이즈 제거, 최우수 |
| Item-KNN | 0.943 ± 0.004 | 0.741 ± 0.003 | 아이템 유사도가 안정적 |
| User-KNN | 0.955 ± 0.005 | 0.754 ± 0.003 | 사용자 취향 변동으로 가장 높은 오차 |

핵심 코드 구조:
```python
# surprise 라이브러리: 추천 시스템 전용 도구
svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
cv_results = cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5)
```

SVD가 최우수인 이유: 93.7% 희소 행렬에서 50차원 잠재 공간으로 압축하면 노이즈가 줄고 일반화 성능이 향상된다. Item-KNN이 User-KNN보다 나은 이유: 영화 간 유사도는 시간에 따라 변하지 않지만, 사용자 취향은 변할 수 있다.

### 프롬프트 6 모범 구현: SVD 잠재 요인 수 변경

```python
from surprise import Dataset, SVD
from surprise.model_selection import cross_validate

data = Dataset.load_builtin("ml-100k")

n_factors_list = [10, 20, 50, 100, 200]

print(f"{'n_factors':<12} {'RMSE':<16} {'MAE':<16}")
print("-" * 44)

for n_factors in n_factors_list:
    svd = SVD(n_factors=n_factors, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    cv = cross_validate(svd, data, measures=["RMSE", "MAE"], cv=5, verbose=False)

    rmse_mean = cv["test_rmse"].mean()
    rmse_std = cv["test_rmse"].std()
    mae_mean = cv["test_mae"].mean()
    mae_std = cv["test_mae"].std()

    print(f"{n_factors:<12} {rmse_mean:.4f} +/- {rmse_std:.4f}  {mae_mean:.4f} +/- {mae_std:.4f}")
```

기대 결과:
- n_factors=10: 표현력 부족으로 RMSE가 상대적으로 높음 (언더피팅)
- n_factors=50: 적절한 복잡도로 최적 성능 부근
- n_factors=100~200: 성능 향상이 둔화되거나 약간 악화 (과적합 경향)
- 핵심: 잠재 요인 수가 많다고 항상 좋은 것이 아니다. 데이터 크기에 맞는 복잡도 선택이 중요

### 프롬프트 7 모범 구현: 특정 사용자 추천 결과 분석

```python
from surprise import Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import pandas as pd

data = Dataset.load_builtin("ml-100k")
trainset = data.build_full_trainset()

svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
svd.fit(trainset)

# 아이템 이름 매핑 (MovieLens 100K)
try:
    from pathlib import Path
    import os
    data_dir = os.path.expanduser("~/.surprise_data/ml-100k/ml-100k")
    item_df = pd.read_csv(f"{data_dir}/u.item", sep="|", encoding="latin-1",
                           header=None, usecols=[0, 1], names=["item_id", "title"])
    item_names = dict(zip(item_df["item_id"].astype(str), item_df["title"]))
except:
    item_names = {}

sample_users = ["1", "50", "100"]

for user_id in sample_users:
    print(f"\n{'='*60}")
    print(f"사용자 {user_id}")
    print(f"{'='*60}")

    # 기존 높은 평점 아이템
    user_inner = trainset.to_inner_uid(user_id)
    user_ratings = [(trainset.to_raw_iid(iid), rating)
                     for iid, rating in trainset.ur[user_inner]]
    high_rated = sorted(user_ratings, key=lambda x: x[1], reverse=True)[:5]

    print("\n기존 높은 평점 (Top 5):")
    for item_id, rating in high_rated:
        name = item_names.get(item_id, f"Item {item_id}")
        print(f"  {name}: {rating:.1f}점")

    # Top-5 추천
    rated_items = set(iid for iid, _ in trainset.ur[user_inner])
    predictions = []
    for iid in range(trainset.n_items):
        if iid not in rated_items:
            raw_iid = trainset.to_raw_iid(iid)
            pred = svd.predict(user_id, raw_iid)
            predictions.append((raw_iid, pred.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    print("\nSVD 추천 (Top 5):")
    for item_id, score in predictions[:5]:
        name = item_names.get(item_id, f"Item {item_id}")
        print(f"  {name}: 예측 {score:.2f}점")
```

기대 결과:
- 추천된 영화가 사용자의 기존 선호(장르, 스타일)와 일관되는 경향
- SVD의 잠재 요인이 사용자 취향 패턴을 학습한 결과
- Cold-start 문제: 평점이 없는 신규 사용자에게는 잠재 벡터를 학습할 데이터가 없어 추천 불가

### 프롬프트 8 모범 구현: User-KNN 이웃 수(k) 변경

```python
from surprise import Dataset, KNNWithMeans
from surprise.model_selection import cross_validate

data = Dataset.load_builtin("ml-100k")

k_values = [5, 10, 20, 40, 80]

print(f"{'k':<8} {'RMSE':<16} {'MAE':<16}")
print("-" * 40)

for k in k_values:
    knn = KNNWithMeans(k=k, sim_options={"name": "cosine", "user_based": True})
    cv = cross_validate(knn, data, measures=["RMSE", "MAE"], cv=5, verbose=False)

    rmse_mean = cv["test_rmse"].mean()
    rmse_std = cv["test_rmse"].std()
    mae_mean = cv["test_mae"].mean()
    mae_std = cv["test_mae"].std()

    print(f"{k:<8} {rmse_mean:.4f} +/- {rmse_std:.4f}  {mae_mean:.4f} +/- {mae_std:.4f}")
```

기대 결과:
- k=5: 이웃이 너무 적어 불안정, RMSE 높음
- k=20~40: 적절한 이웃 수로 최적 성능 부근
- k=80: 너무 많은 이웃이 포함되어 노이즈 증가, 성능 약간 악화
- 핵심: k가 너무 작으면 노이즈에 민감하고, 너무 크면 무관한 사용자까지 포함하여 예측이 평균화됨

---

## 11장 전체 핵심 정리

```text
1. 중심성: 같은 네트워크에서도 지표에 따라 "중요한 노드"가 달라진다.
   연결 중심성(인기), 매개 중심성(중개), 근접 중심성(접근성), PageRank(재귀적 영향력).
2. 커뮤니티: Louvain은 모듈성을 최대화하며 커뮤니티 수를 자동 결정한다.
   해상도 파라미터로 커뮤니티 크기를 조절하며, Q ≥ 0.3이면 유의미하다.
3. 링크 예측: 공통 이웃, Jaccard, Adamic-Adar 등 구조적 유사도로 미래 연결을 추정한다.
   AUC-ROC로 성능을 평가하며, 0.5가 무작위, 1.0이 완벽이다.
4. Node2Vec: 랜덤 워크 + Skip-gram으로 노드를 벡터로 변환한다.
   임베딩 공간에서 가까운 노드가 실제 네트워크에서도 유사한 위치에 있다.
5. 추천: SVD가 KNN보다 희소 데이터에서 우수하다.
   잠재 요인 수(n_factors)와 이웃 수(k)는 과적합과 트레이드오프 관계에 있다.
6. AI 도구로 코드를 생성하되, 파라미터 변경의 의미를 이해하고 결과를 해석하는 것이 핵심이다.
```
