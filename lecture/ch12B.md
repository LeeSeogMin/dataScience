# 12장 B: 그래프 신경망과 지식 그래프 — 모범 답안과 해설

> 이 문서는 실습 제출 후 공개한다. 제출 전에는 열람하지 않는다.

---

## 실습 1 해설: GNN 노드 분류 — Cora 논문 분류

### 제공 코드 실행 결과 해설

Cora 데이터셋(2,708 노드, 10,556 엣지, 1,433차원 특성, 7클래스)에 GCN, GAT, GraphSAGE를 적용한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| GCN Test Acc | 0.80~0.82 | 정규화된 평균 집계가 동질적 이웃에서 효과적 |
| GAT Test Acc | 0.77~0.79 | 어텐션 파라미터가 많아 소규모 데이터에서 약간 과적합 |
| GraphSAGE Test Acc | 0.80~0.82 | 이웃 샘플링+집계가 소규모에서도 안정적 |
| Train Acc (공통) | ~1.000 | 세 모델 모두 학습 데이터는 완벽히 학습 |

핵심 코드 구조:

```python
# GCN 2층 구조: 메시지 패싱 2회 → 2-홉 이웃 정보 집계
self.conv1 = GCNConv(num_features, hidden_dim)  # 1433 → 64
self.conv2 = GCNConv(hidden_dim, num_classes)    # 64 → 7

# GAT: Multi-head attention (8 heads × 8 dim = 64) → Single-head
self.conv1 = GATConv(num_features, hidden_dim=8, heads=8)  # → 64
self.conv2 = GATConv(64, num_classes, heads=1)               # → 7
```

GCN과 GraphSAGE가 비슷한 이유: Cora는 소규모(2,708 노드)이고 같은 주제의 논문은 서로를 인용하는 동질적 구조다. 이런 환경에서는 단순 평균(GCN)과 샘플링 기반 집계(GraphSAGE) 모두 효과적이다. GraphSAGE의 진정한 강점은 수백만 노드 이상의 대규모 그래프에서 발휘된다.

### 프롬프트 1 모범 구현: GNN 레이어 수에 따른 오버스무딩 관찰

```python
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from sklearn.metrics import accuracy_score

torch.manual_seed(42)
np.random.seed(42)

dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

class FlexibleGCN(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.convs.append(GCNConv(num_features, num_classes))
        else:
            self.convs.append(GCNConv(num_features, hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.convs.append(GCNConv(hidden_dim, num_classes))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

print(f"{'레이어 수':<12} {'Test Acc':<12}")
print("-" * 24)

for num_layers in [2, 3, 4, 6, 8]:
    torch.manual_seed(42)
    model = FlexibleGCN(data.num_features, dataset.num_classes, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0
    best_state = None

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(data.x, data.edge_index).argmax(dim=1)
            val_acc = accuracy_score(data.y[data.val_mask].cpu(), pred[data.val_mask].cpu())
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = model.state_dict().copy()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        test_acc = accuracy_score(data.y[data.test_mask].cpu(), pred[data.test_mask].cpu())

    print(f"{num_layers:<12} {test_acc:<12.4f}")
```

기대 결과 해석:
- 2~3층: Test Acc 0.80~0.82. 2-홉~3-홉 이웃 정보가 충분
- 4층: 약간 하락 시작. 오버스무딩의 초기 징후
- 6~8층: 뚜렷한 성능 하락 (0.60~0.75). 노드 표현이 평균으로 수렴
- 핵심: 레이어를 깊이 쌓는다고 좋아지지 않는다. Cora에서는 2-3층이 최적이다

---

## 실습 2 해설: GNN 링크 예측 — GraphSAGE vs 전통 방법

### 제공 코드 실행 결과 해설

Cora 데이터셋에서 엣지의 10%를 테스트셋으로 분할하고, GraphSAGE와 전통적 방법의 링크 예측 성능을 비교한 결과:

| 방법 | AUC 경향 | AP 경향 | 이유 |
| ---- | -------- | ------- | ---- |
| Common Neighbors | 0.69~0.71 | 0.69~0.70 | 공통 이웃 수만 사용 |
| Jaccard | 0.69~0.70 | 0.69~0.70 | 공통/전체 이웃 비율 |
| Adamic-Adar | 0.69~0.71 | 0.70~0.71 | 희귀 이웃에 가중치 |
| GraphSAGE | 0.74~0.76 | 0.73~0.75 | 노드 특성 + 다중 홉 학습 |

GraphSAGE가 전통 방법 대비 약 7% AUC 향상을 보인 이유: 전통 방법은 그래프 구조(연결 패턴)만 사용하지만, GraphSAGE는 Cora의 1,433차원 단어 벡터(노드 특성)까지 함께 학습한다. 공통 이웃이 없어도 비슷한 단어를 사용하는 논문 간의 잠재적 연결을 예측할 수 있다.

### 프롬프트 2 모범 구현: 디코더 방식 비교 — 내적 vs MLP

```python
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from torch_geometric.transforms import RandomLinkSplit
from sklearn.metrics import roc_auc_score, average_precision_score

torch.manual_seed(42)
np.random.seed(42)

dataset = Planetoid(root='./data', name='Cora')
data = dataset[0]

transform = RandomLinkSplit(
    num_val=0.05, num_test=0.10, is_undirected=True,
    add_negative_train_samples=True, neg_sampling_ratio=1.0
)
train_data, val_data, test_data = transform(data)

class GraphSAGE(torch.nn.Module):
    def __init__(self, num_features, hidden_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)

class DotProductDecoder(torch.nn.Module):
    def forward(self, x_i, x_j):
        return torch.sigmoid((x_i * x_j).sum(dim=-1))

class MLPDecoder(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim=64):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x_i, x_j):
        x = torch.cat([x_i, x_j], dim=-1)
        x = F.relu(self.lin1(x))
        return torch.sigmoid(self.lin2(x)).squeeze()

def train_and_evaluate(encoder, decoder, train_data, test_data, epochs=100):
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()), lr=0.01
    )
    for epoch in range(epochs):
        encoder.train(); decoder.train()
        optimizer.zero_grad()
        z = encoder(train_data.x, train_data.edge_index)
        pos_edge = train_data.edge_label_index[:, train_data.edge_label == 1]
        neg_edge = train_data.edge_label_index[:, train_data.edge_label == 0]
        pos_pred = decoder(z[pos_edge[0]], z[pos_edge[1]])
        neg_pred = decoder(z[neg_edge[0]], z[neg_edge[1]])
        loss = F.binary_cross_entropy(pos_pred, torch.ones_like(pos_pred)) + \
               F.binary_cross_entropy(neg_pred, torch.zeros_like(neg_pred))
        loss.backward(); optimizer.step()

    encoder.eval(); decoder.eval()
    with torch.no_grad():
        z = encoder(test_data.x, test_data.edge_index)
        pos_edge = test_data.edge_label_index[:, test_data.edge_label == 1]
        neg_edge = test_data.edge_label_index[:, test_data.edge_label == 0]
        pos_pred = decoder(z[pos_edge[0]], z[pos_edge[1]]).cpu().numpy()
        neg_pred = decoder(z[neg_edge[0]], z[neg_edge[1]]).cpu().numpy()
    y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    y_score = np.concatenate([pos_pred, neg_pred])
    return roc_auc_score(y_true, y_score), average_precision_score(y_true, y_score)

print(f"{'디코더':<16} {'AUC':<10} {'AP':<10}")
print("-" * 36)

for name, DecoderClass in [("Dot Product", DotProductDecoder), ("MLP", MLPDecoder)]:
    torch.manual_seed(42)
    encoder = GraphSAGE(data.num_features)
    decoder = DecoderClass() if name == "Dot Product" else DecoderClass(32)
    auc, ap = train_and_evaluate(encoder, decoder, train_data, test_data)
    print(f"{name:<16} {auc:<10.4f} {ap:<10.4f}")
```

기대 결과:
- MLP 디코더가 Dot Product보다 AUC/AP가 약간 높을 가능성이 있음
- Dot Product: 파라미터가 없어 과적합 위험이 낮고 계산이 빠름. 대규모 추론에 유리
- MLP: 파라미터가 있어 "비슷한 노드 연결" 외에 상보적 관계도 학습 가능. 데이터가 충분할 때 유리
- 실무에서는 Dot Product로 시작하고, 성능이 부족하면 MLP를 시도하는 것이 권장됨

---

## 실습 3 해설: LightGCN 영화 추천

### 제공 코드 실행 결과 해설

MovieLens 100K 데이터(943 사용자, ~1,682 아이템, 100,000 평점)에서 LightGCN을 학습한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| Recall@10 | 0.08~0.10 | 희소한 데이터에서 Top-10 정답 포착 비율 |
| NDCG@10 | 0.10~0.13 | 정답이 상위에 배치되는 정도 |
| SVD RMSE (11장) | ~0.935 | 평점 예측 정확도 (비교 불가한 다른 지표) |

핵심 코드 구조:

```python
# LightGCN: 비선형 변환 없이 정규화된 이웃 집계만 수행
for _ in range(self.num_layers):
    out = torch.zeros_like(all_emb)
    out.index_add_(0, col, all_emb[row] * norm.unsqueeze(1))
    all_emb = out
    embs.append(all_emb)

# Layer combination: 모든 층의 평균
final_emb = torch.stack(embs, dim=0).mean(dim=0)

# BPR 손실: 본 영화 > 안 본 영화 순서 학습
loss = -F.logsigmoid(pos_scores - neg_scores).mean()
```

SVD와 LightGCN을 직접 비교할 수 없는 이유: SVD는 평점의 절대값 예측(RMSE 최소화), LightGCN은 상대 순위 최적화(BPR). 같은 데이터지만 다른 문제를 풀고 있다. Top-N 추천이 목적이라면 LightGCN이, 평점 예측이 목적이라면 SVD가 적합하다.

### 프롬프트 3 모범 구현: LightGCN 레이어 수 실험

```python
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from sklearn.model_selection import train_test_split
from collections import defaultdict
import urllib.request
import zipfile

torch.manual_seed(42)
np.random.seed(42)

INPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'input')
os.makedirs(INPUT_DIR, exist_ok=True)

# MovieLens 100K 로드 (12-4-lightgcn.py와 동일한 전처리)
data_file = os.path.join(INPUT_DIR, 'ml-100k', 'u.data')
if not os.path.exists(data_file):
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = os.path.join(INPUT_DIR, 'ml-100k.zip')
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(INPUT_DIR)
    os.remove(zip_path)

df = pd.read_csv(data_file, sep='\t', header=None, names=['user_id', 'item_id', 'rating', 'timestamp'])
user_mapping = {u: i for i, u in enumerate(df['user_id'].unique())}
item_mapping = {i: j for j, i in enumerate(df['item_id'].unique())}
df['user'] = df['user_id'].map(user_mapping)
df['item'] = df['item_id'].map(item_mapping)
num_users = len(user_mapping)
num_items = len(item_mapping)

df_pos = df[df['rating'] >= 4].sort_values('timestamp')
train_df, test_df = train_test_split(df_pos, test_size=0.2, shuffle=False)

train_user = torch.LongTensor(train_df['user'].values)
train_item = torch.LongTensor(train_df['item'].values) + num_users
edge_index = torch.stack([
    torch.cat([train_user, train_item]),
    torch.cat([train_item, train_user])
])

train_user_items = defaultdict(set)
for _, row in train_df.iterrows():
    train_user_items[row['user']].add(row['item'])
test_user_items = defaultdict(set)
for _, row in test_df.iterrows():
    test_user_items[row['user']].add(row['item'])

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, num_layers=3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_layers = num_layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)

    def forward(self, edge_index):
        user_emb = self.user_embedding.weight
        item_emb = self.item_embedding.weight
        all_emb = torch.cat([user_emb, item_emb], dim=0)
        row, col = edge_index
        deg = degree(col, all_emb.size(0), dtype=all_emb.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        embs = [all_emb]
        for _ in range(self.num_layers):
            out = torch.zeros_like(all_emb)
            out.index_add_(0, col, all_emb[row] * norm.unsqueeze(1))
            all_emb = out
            embs.append(all_emb)
        final_emb = torch.stack(embs, dim=0).mean(dim=0)
        return final_emb[:self.num_users], final_emb[self.num_users:]

    def predict(self, user_ids, item_ids, user_emb, item_emb):
        return (user_emb[user_ids] * item_emb[item_ids]).sum(dim=1)

def recall_at_k(actual, predicted, k=10):
    predicted = predicted[:k]
    return len(set(actual) & set(predicted)) / len(actual) if actual else 0.0

def ndcg_at_k(actual, predicted, k=10):
    predicted = predicted[:k]
    dcg = sum(1.0 / np.log2(i + 2) for i, item in enumerate(predicted) if item in actual)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg > 0 else 0.0

all_items = set(range(num_items))

print(f"{'num_layers':<14} {'Recall@10':<12} {'NDCG@10':<12}")
print("-" * 38)

for n_layers in [1, 2, 3, 4, 5]:
    torch.manual_seed(42)
    model = LightGCN(num_users, num_items, embedding_dim=64, num_layers=n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        model.train()
        perm = torch.randperm(len(train_df))[:1024]
        batch_users = torch.LongTensor(train_df.iloc[perm]['user'].values)
        batch_pos = torch.LongTensor(train_df.iloc[perm]['item'].values)
        batch_neg = torch.LongTensor([
            np.random.choice(list(all_items - train_user_items[u]))
            for u in batch_users.numpy()
        ])
        user_emb, item_emb = model(edge_index)
        pos_scores = model.predict(batch_users, batch_pos, user_emb, item_emb)
        neg_scores = model.predict(batch_users, batch_neg, user_emb, item_emb)
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        reg = 0.0001 * (model.user_embedding.weight[batch_users].norm(2).pow(2) +
                        model.item_embedding.weight[batch_pos].norm(2).pow(2) +
                        model.item_embedding.weight[batch_neg].norm(2).pow(2)) / 1024
        (loss + reg).backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(edge_index)
    recalls, ndcgs = [], []
    test_users = list(test_user_items.keys())[:500]
    for user in test_users:
        if user not in train_user_items: continue
        scores = (user_emb[user] * item_emb).sum(dim=1).cpu().numpy()
        for item in train_user_items[user]: scores[item] = -np.inf
        top_k = np.argsort(scores)[-10:][::-1].tolist()
        actual = list(test_user_items[user])
        recalls.append(recall_at_k(actual, top_k))
        ndcgs.append(ndcg_at_k(actual, top_k))

    print(f"{n_layers:<14} {np.mean(recalls):<12.4f} {np.mean(ndcgs):<12.4f}")
```

기대 결과:
- 1층: 직접 상호작용만 반영. 행렬 분해와 유사한 수준
- 2~3층: 성능 향상. 고차 연결 패턴(친구의 친구가 본 영화) 학습
- 4~5층: 3층과 비슷하거나 약간 하락. Layer combination 덕분에 오버스무딩이 완화됨
- 핵심: LightGCN의 layer combination(모든 층 평균)은 오버스무딩에 대한 자연스러운 방어 메커니즘이다. 노드 분류(GCN)보다 성능 하락이 덜하다

### 프롬프트 4 모범 구현: embedding_dim 변경 실험

```python
# 위 코드에서 num_layers 루프를 embedding_dim 루프로 변경
print(f"{'embedding_dim':<16} {'Recall@10':<12} {'NDCG@10':<12}")
print("-" * 40)

for emb_dim in [16, 32, 64, 128]:
    torch.manual_seed(42)
    model = LightGCN(num_users, num_items, embedding_dim=emb_dim, num_layers=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # ... (학습 및 평가 코드는 위와 동일, embedding_dim만 변경) ...
    # 결과 출력
    print(f"{emb_dim:<16} {np.mean(recalls):<12.4f} {np.mean(ndcgs):<12.4f}")
```

기대 결과:
- 16차원: 표현력 부족. 사용자/아이템의 다양한 선호를 담기 어려움
- 32~64차원: 적절한 표현력. MovieLens 100K 규모에 적합
- 128차원: 과적합 가능성. 데이터 크기 대비 파라미터가 많음
- 핵심: 차원이 크다고 항상 좋은 것이 아니다. 데이터 규모에 맞는 차원 선택이 중요

---

## 실습 4 해설: 지식 그래프 임베딩 — TransE vs RotatE

### 제공 코드 실행 결과 해설

FB15k-237 데이터셋(14,541 엔티티, 237 관계)에서 TransE와 RotatE의 링크 예측 결과:

| 모델 | MRR 경향 | Hits@1 경향 | Hits@10 경향 | 이유 |
| ---- | -------- | ----------- | ------------ | ---- |
| TransE | 0.15~0.17 | 0.08~0.10 | 0.28~0.32 | 이동 모델: 대칭/역 관계 한계 |
| RotatE | 0.22~0.25 | 0.14~0.17 | 0.38~0.42 | 회전 모델: 대칭/역 관계 처리 가능 |

RotatE가 우수한 이유: FB15k-237은 Freebase에서 추출되어 대칭 관계(결혼, 형제), 역 관계(부모↔자녀), 합성 관계(부+형=삼촌)가 다수 포함되어 있다. TransE의 h + r = t 모델은 대칭 관계에서 r = 0이 되어야 하는 문제가 있지만, RotatE는 180도 회전으로 자연스럽게 표현한다.

### 프롬프트 5 모범 구현: embedding_dim 변경 실험

```python
import torch
import numpy as np
from pykeen.pipeline import pipeline
from pykeen.datasets import FB15k237

torch.manual_seed(42)
np.random.seed(42)

dataset = FB15k237()

print(f"{'embedding_dim':<16} {'MRR':<10} {'Hits@10':<10}")
print("-" * 36)

for emb_dim in [50, 100, 200]:
    result = pipeline(
        dataset=dataset,
        model='TransE',
        model_kwargs={'embedding_dim': emb_dim},
        training_kwargs={'num_epochs': 100, 'batch_size': 256},
        optimizer='Adam',
        optimizer_kwargs={'lr': 0.001},
        negative_sampler='basic',
        negative_sampler_kwargs={'num_negs_per_pos': 1},
        evaluator_kwargs={'batch_size': 256},
        random_seed=42
    )

    metrics = result.metric_results.to_dict()
    mrr = metrics['both']['realistic']['inverse_harmonic_mean_rank']
    hits10 = metrics['both']['realistic']['hits_at_10']

    print(f"{emb_dim:<16} {mrr:<10.4f} {hits10:<10.4f}")
```

기대 결과:
- 50차원: 표현력 부족. 14,541 엔티티와 237 관계를 50차원에 표현하기 어려움
- 100차원: 적절한 표현력. 기본 설정
- 200차원: 약간 개선되거나 비슷. 100 에폭으로는 큰 차원의 장점을 충분히 살리기 어려움
- 핵심: 차원을 늘리면 표현력은 높아지지만, 학습 데이터와 에폭이 충분해야 효과를 볼 수 있다

---

## 실습 5 해설: GraphRAG 개념 실습

### 제공 코드 실행 결과 해설

물리학 선구자 지식 그래프(10 엔티티, 13 관계)로 Vector RAG와 GraphRAG를 비교한 결과:

| 질의 | Vector RAG | GraphRAG |
| ---- | ---------- | -------- |
| Einstein과 quantum mechanics 관계? | 키워드 매칭 문서만 반환 | Einstein → Bohr → Quantum Mechanics 경로 발견 |
| Nobel Prize 수상자? | "Nobel Prize" 포함 문서 1-2개 | Einstein, Bohr, Planck 모두 탐색 |

핵심 차이:
- Vector RAG는 질의와 문서의 키워드 겹침에 의존. "Einstein"과 "quantum mechanics"가 같은 문서에 없으면 관계를 찾지 못함
- GraphRAG는 엔티티 간 경로를 탐색하여 간접적 관계(Einstein → debated_with → Bohr → contributed_to → Quantum Mechanics)도 발견
- 커뮤니티 탐지로 "물리학자 그룹", "이론 그룹" 같은 전역 요약도 가능

### 프롬프트 6 모범 구현: 새로운 엔티티와 관계 추가

```python
import networkx as nx
from collections import defaultdict

# 기존 지식 그래프에 새 엔티티/관계 추가
EXTENDED_KG = {
    "entities": [
        {"id": "e1", "name": "Albert Einstein", "type": "Person"},
        {"id": "e2", "name": "Theory of Relativity", "type": "Theory"},
        {"id": "e3", "name": "Niels Bohr", "type": "Person"},
        {"id": "e4", "name": "Quantum Mechanics", "type": "Theory"},
        {"id": "e5", "name": "Max Planck", "type": "Person"},
        {"id": "e6", "name": "Planck Constant", "type": "Concept"},
        {"id": "e7", "name": "Princeton University", "type": "Organization"},
        {"id": "e8", "name": "Copenhagen Interpretation", "type": "Theory"},
        {"id": "e9", "name": "Nobel Prize in Physics", "type": "Award"},
        {"id": "e10", "name": "Photoelectric Effect", "type": "Phenomenon"},
        # 새로 추가
        {"id": "e11", "name": "Werner Heisenberg", "type": "Person"},
        {"id": "e12", "name": "Uncertainty Principle", "type": "Theory"},
    ],
    "relations": [
        {"source": "e1", "target": "e2", "type": "developed"},
        {"source": "e1", "target": "e10", "type": "explained"},
        {"source": "e1", "target": "e9", "type": "received"},
        {"source": "e1", "target": "e7", "type": "worked_at"},
        {"source": "e1", "target": "e3", "type": "debated_with"},
        {"source": "e3", "target": "e4", "type": "contributed_to"},
        {"source": "e3", "target": "e8", "type": "developed"},
        {"source": "e3", "target": "e9", "type": "received"},
        {"source": "e5", "target": "e6", "type": "discovered"},
        {"source": "e5", "target": "e4", "type": "founded"},
        {"source": "e5", "target": "e9", "type": "received"},
        {"source": "e10", "target": "e4", "type": "evidence_for"},
        {"source": "e6", "target": "e4", "type": "fundamental_to"},
        # 새로 추가
        {"source": "e11", "target": "e4", "type": "contributed_to"},
        {"source": "e11", "target": "e12", "type": "developed"},
        {"source": "e11", "target": "e9", "type": "received"},
    ]
}

def build_kg(kg_data):
    G = nx.DiGraph()
    for entity in kg_data["entities"]:
        G.add_node(entity["id"], name=entity["name"], type=entity["type"])
    for rel in kg_data["relations"]:
        G.add_edge(rel["source"], rel["target"], relation=rel["type"])
    return G

G = build_kg(EXTENDED_KG)
print(f"확장된 KG: {G.number_of_nodes()} 엔티티, {G.number_of_edges()} 관계")

# "Who contributed to Quantum Mechanics?" 질의
qm_id = None
for node, attrs in G.nodes(data=True):
    if attrs["name"] == "Quantum Mechanics":
        qm_id = node
        break

print("\n질의: Who contributed to Quantum Mechanics?")
print("결과 (Quantum Mechanics에 연결된 엔티티):")

# 역방향 탐색: Quantum Mechanics를 가리키는 엔티티
for predecessor in G.predecessors(qm_id):
    edge_data = G.get_edge_data(predecessor, qm_id)
    print(f"  - {G.nodes[predecessor]['name']} ({edge_data['relation']})")
```

기대 결과:
- Quantum Mechanics를 가리키는 엔티티: Niels Bohr (contributed_to), Max Planck (founded), Werner Heisenberg (contributed_to), Photoelectric Effect (evidence_for), Planck Constant (fundamental_to)
- Heisenberg 추가로 Quantum Mechanics의 기여자 목록이 확장됨
- GraphRAG의 강점: 새 엔티티를 추가하면 자동으로 관련 질의 결과에 반영됨

### 프롬프트 7 모범 구현: 지식 그래프 통계 분석

```python
import networkx as nx
from collections import defaultdict

# 12-6-graphrag.py의 SAMPLE_KNOWLEDGE_GRAPH를 사용 (또는 위 확장 버전)
# ... (그래프 구축 코드는 원본과 동일) ...

G = build_knowledge_graph(SAMPLE_KNOWLEDGE_GRAPH)

# 각 엔티티의 차수 (방향 무시)
G_undirected = G.to_undirected()
degrees = dict(G_undirected.degree())

# Top-3 연결 많은 엔티티
sorted_nodes = sorted(degrees.items(), key=lambda x: -x[1])
print("연결이 가장 많은 엔티티 Top-3:")
for node, deg in sorted_nodes[:3]:
    print(f"  {G.nodes[node]['name']} (유형: {G.nodes[node]['type']}, 차수: {deg})")

# 엔티티 유형별 평균 차수
type_degrees = defaultdict(list)
for node, deg in degrees.items():
    entity_type = G.nodes[node]['type']
    type_degrees[entity_type].append(deg)

print(f"\n{'엔티티 유형':<20} {'평균 차수':<10} {'엔티티 수':<10}")
print("-" * 40)
for entity_type, degs in sorted(type_degrees.items(), key=lambda x: -sum(x[1])/len(x[1])):
    avg_deg = sum(degs) / len(degs)
    print(f"{entity_type:<20} {avg_deg:<10.1f} {len(degs):<10}")
```

기대 결과:
- Top-3: Albert Einstein (차수 5), Quantum Mechanics (차수 4), Nobel Prize in Physics (차수 3)
- Person 유형의 평균 차수가 가장 높음: 물리학자들이 여러 이론, 상, 기관과 연결
- 높은 차수의 엔티티는 지식 그래프에서 "허브" 역할. GraphRAG에서 이런 허브를 경유하면 다양한 질의에 답할 수 있다

### 프롬프트 8 모범 구현: TF-IDF 기반 Vector RAG 개선

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

documents = [
    "Albert Einstein developed the Theory of Relativity.",
    "Niels Bohr contributed to Quantum Mechanics.",
    "Max Planck discovered the Planck Constant.",
    "Einstein and Bohr debated about quantum mechanics.",
    "The photoelectric effect provided evidence for quantum mechanics."
]

test_queries = [
    "What is the connection between Einstein and quantum mechanics?",
    "Who received the Nobel Prize?",
    "What theories did Einstein develop?"
]

# 기존 키워드 매칭
def keyword_matching(query, docs):
    query_terms = set(query.lower().split())
    scored = []
    for doc in docs:
        doc_terms = set(doc.lower().split())
        score = len(query_terms & doc_terms) / len(query_terms) if query_terms else 0
        scored.append((doc, score))
    return sorted(scored, key=lambda x: -x[1])[:3]

# TF-IDF 코사인 유사도
def tfidf_search(query, docs):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(docs)
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    ranked = sorted(zip(docs, similarities), key=lambda x: -x[1])
    return ranked[:3]

for query in test_queries:
    print(f"\n질의: {query}")

    kw_results = keyword_matching(query, documents)
    print(f"  키워드 매칭 Top-1: {kw_results[0][0][:50]}... (score: {kw_results[0][1]:.2f})")

    tfidf_results = tfidf_search(query, documents)
    print(f"  TF-IDF Top-1:      {tfidf_results[0][0][:50]}... (score: {tfidf_results[0][1]:.2f})")
```

기대 결과:
- TF-IDF가 키워드 매칭보다 정교한 유사도 계산. 단어 빈도와 문서 빈도를 고려
- 그러나 두 방법 모두 문서 내 키워드에 의존하는 한계는 동일
- "Einstein과 quantum mechanics의 연결"은 두 키워드가 같은 문서에 있어야 검색됨
- 이 한계를 극복하는 것이 GraphRAG의 다중 홉 탐색

---

## 12장 전체 핵심 정리

```text
1. GNN 오버스무딩: 레이어를 깊이 쌓으면 성능이 오히려 하락한다.
   Cora에서 2-3층이 최적이며, 8층에서는 뚜렷한 성능 저하가 관찰된다.
2. GNN 아키텍처: 소규모+동질적 이웃은 GCN, 이웃 중요도 차이가 크면 GAT,
   대규모+신규 노드 유입은 GraphSAGE를 선택한다.
3. 링크 예측: GraphSAGE가 전통적 방법(CN, Jaccard) 대비 ~7% AUC 향상.
   노드 특성과 다중 홉 정보를 함께 학습하는 것이 핵심이다.
4. LightGCN: 추천에서는 단순한 구조가 효과적이다. Layer combination으로
   오버스무딩이 완화되어 3-4층도 안정적이다.
5. KGE: RotatE는 TransE 대비 MRR에서 ~7%p 향상. 대칭/역 관계가 많은
   데이터에서 복소수 회전 모델링이 유리하다.
6. GraphRAG: Vector RAG의 단일 홉 한계를 그래프 탐색으로 극복한다.
   다중 홉 추론과 전역 요약이 필요한 상황에서 효과적이다.
7. 실무 전략: 전통적 방법으로 베이스라인을 설정하고, GNN으로 개선 여부를
   확인한다. 전통적 방법이 충분하면 복잡한 GNN이 불필요할 수 있다.
```
