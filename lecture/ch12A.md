## 12주차: 그래프 신경망과 지식 그래프 — 이론과 실습

> **미션**: GNN의 메시지 패싱 원리를 이해하고, 노드 분류·링크 예측·추천·지식 그래프 임베딩·GraphRAG를 직접 구현하며, 전통적 그래프 분석과 딥러닝 기반 분석의 차이를 설명할 수 있다

### 학습목표

이 수업을 마치면 다음을 수행할 수 있다:

1. GNN의 메시지 패싱(생성→집계→갱신)을 설명하고, 오버스무딩 문제와 해결 전략을 이해한다
2. GCN, GAT, GraphSAGE의 차이를 비교하고, 상황에 맞는 GNN을 선택할 수 있다
3. GNN을 이용한 노드 분류와 링크 예측을 수행하고, 전통적 방법과 성능을 비교할 수 있다
4. LightGCN의 단순화 원리와 BPR 손실 함수를 이해하고, 추천 시스템에 적용할 수 있다
5. TransE와 RotatE의 관계 모델링 차이를 설명하고, 지식 그래프 링크 예측을 수행할 수 있다
6. GraphRAG의 개념과 Vector RAG 대비 장점을 이해하고, 지식 그래프 기반 질의응답을 실습할 수 있다

### 실습 방식

실습은 **실행 → 이해 → 직접 코딩** 3단계로 진행한다.

1. **실행**: 제공된 코드를 그대로 실행해 결과를 확인한다
2. **이해**: 코드 구조와 결과를 읽고 왜 그런 결과가 나왔는지 파악한다
3. **직접 코딩**: AI 코딩 도구(Copilot, Claude, ChatGPT 등)에 프롬프트를 주어 코드를 수정하거나 새로 작성한다

**제출 형태**: 개별 제출 — 실행 결과 + 직접 작성한 코드 + 해석

**실습 환경 준비**:

```bash
pip install torch torch-geometric networkx scikit-learn matplotlib numpy pandas
pip install pykeen  # 실습 4 (지식 그래프 임베딩)
```

---

### 12.1 GNN의 원리: 메시지 패싱

CNN은 이미지(2D 격자), RNN은 시퀀스(1D 배열)처럼 규칙적인 구조를 가정한다. 그래프는 노드마다 이웃 수가 다르고 순서도 없어 이런 모델을 직접 적용할 수 없다. GNN(Graph Neural Network)은 **"이웃의 정보를 모아서 나를 업데이트한다"**는 아이디어로 그래프에서 직접 패턴을 학습한다.

비유하면, 동네 주민들이 이웃과 수다를 떨며 동네 소식을 파악하는 과정과 같다. 처음에는 자기 집 앞 일만 알지만, 이웃과 대화할수록 마을 전체 상황을 이해하게 된다.

#### 메시지 패싱 3단계

| 단계 | 이름 | 하는 일 | 비유 |
| ---- | ---- | ------- | ---- |
| 1단계 | 메시지 생성 | 이웃이 자기 정보를 담은 메시지를 보낸다 | "나는 이런 사람이야" |
| 2단계 | 집계 | 여러 이웃 메시지를 하나로 합친다 (합/평균/최대) | "이웃들 얘기를 종합하면..." |
| 3단계 | 갱신 | 이웃 정보 + 내 정보를 합쳐 새 표현을 만든다 | "종합해보니 나는 이런 위치구나" |

메시지 패싱을 1번 하면 직접 이웃(1-홉), 2번 하면 이웃의 이웃(2-홉) 정보가 반영된다. 소셜 네트워크에서 2층 GNN은 "친구의 친구"까지 고려해 사용자를 표현한다.

![GNN 메시지 패싱 프레임워크](../diagram/ch12_message_passing.png)

#### 오버스무딩: 층을 너무 깊이 쌓으면?

GNN 층을 많이 쌓으면(10층 이상) 모든 노드 표현이 그래프 평균으로 수렴해 **구별이 불가능**해진다. 이미지에 블러를 반복 적용하면 모두 비슷해지는 것과 같다. 실무에서는 **2~4층**으로 시작하고, 잔차 연결(ResNet처럼 이전 층 출력을 더함)이나 레이어 조합(모든 층의 평균 사용)으로 완화한다.

---

### 12.2 주요 GNN 아키텍처: GCN, GAT, GraphSAGE

| 특성 | GCN | GAT | GraphSAGE |
| ---- | --- | --- | --------- |
| 핵심 아이디어 | 이웃 평균 + 정규화 | 이웃마다 중요도(어텐션)를 학습 | 이웃 일부만 샘플링 + 다양한 집계 |
| 이웃 가중치 | 차수 기반 고정 | 학습으로 결정 | 고정 또는 학습 |
| 대규모 그래프 | 중간 | 낮음 | 높음 (미니배치 가능) |
| 새 노드 처리 | 불가 (변환적) | 불가 (변환적) | 가능 (귀납적) |
| 적합 상황 | 소규모, 동질적 이웃 | 이웃 중요도 다를 때 | 대규모, 신규 노드 유입 |

**GCN**: 자신과 이웃의 특성을 차수로 정규화한 평균 → 가중치 행렬 곱 → 활성화. 가장 단순하고 빠르다.

**GAT**: "모든 이웃이 똑같이 중요하지 않다." 어텐션 메커니즘으로 이웃별 중요도를 학습한다. 논문 인용 네트워크에서 "이 논문은 주제가 비슷하니 중요도 0.7, 저 논문은 0.1"처럼 자동으로 결정한다.

**GraphSAGE**: 이웃 전체가 아닌 일부만 샘플링해 계산량을 통제한다. 팔로워 100만 명인 인플루언서도 10명만 샘플링해 처리 가능. Pinterest가 수십억 핀 추천에 이 방식을 사용했다.

![GraphSAGE 동작 원리](../diagram/ch12_graphsage.png)

---

### 🔬 실습 1: GNN 노드 분류 — Cora 논문 분류

#### Step 1 — 실행

`practice/chapter12/code/12-2-gnn-node-classification.py`를 실행한다.

```bash
cd practice/chapter12/code
python 12-2-gnn-node-classification.py
```

출력에서 아래 표를 채운다.

**데이터셋 정보**:

| 항목 | 값 |
| ---- | -- |
| 노드 수 | |
| 엣지 수 | |
| 특성 차원 | |
| 클래스 수 | |

**모델별 성능 비교**:

| 모델 | Train Acc | Val Acc | Test Acc |
| ---- | --------- | ------- | -------- |
| GCN | | | |
| GAT | | | |
| GraphSAGE | | | |

#### Step 2 — 이해

코드의 핵심 구조를 확인한다.

```python
# GCN: 2층 메시지 패싱으로 2-홉 이웃 정보를 집계
self.conv1 = GCNConv(num_features, hidden_dim)  # 입력 → 64차원
self.conv2 = GCNConv(hidden_dim, num_classes)    # 64차원 → 7클래스
```

- GCN과 GraphSAGE의 Test Acc가 비슷한 이유는? (힌트: Cora는 소규모이고 이웃이 동질적)
- GAT가 약간 낮은 이유는? (힌트: 어텐션 파라미터가 많아 소규모 데이터에서 과적합 가능)
- 저장된 t-SNE 시각화에서 같은 색(같은 주제) 논문이 가까이 모여 있는가? 이것이 GNN이 인용 관계를 통해 주제를 학습했다는 증거다

![GNN 노드 임베딩 시각화](../diagram/ch12_gcn_tsne.png)

#### Step 3 — 직접 코딩

**프롬프트 1**: GNN 레이어 수에 따른 오버스무딩 관찰

> `12-2-gnn-node-classification.py`를 수정해서, GCN의 레이어 수를 2, 3, 4, 6, 8로 바꾸며 Test Accuracy를 비교하는 표를 출력하는 코드를 작성해줘. 레이어가 깊어질수록 성능이 어떻게 변하는지 확인하고 싶어.

결과를 기록한다:

| 레이어 수 | Test Acc |
| --------- | -------- |
| 2 | |
| 3 | |
| 4 | |
| 6 | |
| 8 | |

- 레이어가 깊어질수록 성능이 올라가는가, 내려가는가?
- 성능이 떨어지기 시작하는 지점이 오버스무딩의 시작이다. 몇 층부터인가?

---

### 12.3 GNN 링크 예측: 관계 예측

링크 예측은 현재 없는 엣지가 미래에 생길 확률을 추정하는 문제다. 11장의 공통 이웃(CN), Jaccard, Adamic-Adar는 직관적이지만 복잡한 패턴을 놓친다. GNN은 노드 특성과 다중 홉 이웃 정보를 함께 학습해 더 정확한 예측이 가능하다.

**링크 확률 계산 (디코더)**:

| 디코더 | 방식 | 장점 | 단점 |
| ------ | ---- | ---- | ---- |
| 내적 (Dot Product) | 두 임베딩의 내적 → 시그모이드 | 단순, 빠름 | "비슷하면 연결" 규칙만 표현 |
| MLP | 두 임베딩을 이어붙여 신경망 통과 | 복잡한 관계 학습 가능 | 과적합 위험 |

---

### 🔬 실습 2: GNN 링크 예측 — GraphSAGE vs 전통 방법

#### Step 1 — 실행

`practice/chapter12/code/12-3-gnn-link-prediction.py`를 실행한다.

```bash
python 12-3-gnn-link-prediction.py
```

출력에서 아래 표를 채운다.

**방법별 링크 예측 성능**:

| 방법 | AUC | AP |
| ---- | --- | -- |
| Common Neighbors | | |
| Jaccard | | |
| Adamic-Adar | | |
| GraphSAGE | | |

| 항목 | 값 |
| ---- | -- |
| GraphSAGE AUC 향상률 (vs 최고 전통 방법) | |

#### Step 2 — 이해

코드의 핵심 구조를 확인한다.

```python
# GraphSAGE 인코더: 노드 임베딩 생성
self.conv1 = SAGEConv(num_features, hidden_dim)  # 1433차원 → 64차원
self.conv2 = SAGEConv(hidden_dim, out_dim)        # 64차원 → 32차원

# MLP 디코더: 두 노드 임베딩으로 링크 확률 계산
x = torch.cat([x_i, x_j], dim=-1)  # 32+32 = 64차원
```

- GraphSAGE가 전통 방법보다 AUC가 높은 이유: 노드 특성(1,433차원 단어 벡터)과 다중 홉 이웃 정보를 함께 활용
- 전통 방법은 "공통 이웃이 많으면 연결"이라는 단순 규칙만 사용. 공통 이웃이 없어도 비슷한 주제의 논문을 연결 예측할 수 있는 것이 GNN의 강점

![링크 예측 방법별 성능 비교](../diagram/ch12_link_prediction_comparison.png)

#### Step 3 — 직접 코딩

**프롬프트 2** (선택): 디코더 방식 비교 — 내적 vs MLP

> `12-3-gnn-link-prediction.py`를 수정해서, MLP 디코더 대신 내적(Dot Product) 디코더를 사용한 결과와 비교하는 코드를 작성해줘. 내적 디코더는 `score = sigmoid(sum(z_i * z_j))`로 계산해줘.

| 디코더 | AUC | AP |
| ------ | --- | -- |
| MLP | | |
| Dot Product | | |

- 어떤 디코더가 더 성능이 좋은가?
- 데이터가 적을 때와 많을 때 각각 어떤 디코더가 유리한가?

---

### 12.4 LightGCN: GNN 기반 추천 시스템

행렬 분해(11장 SVD)는 사용자와 아이템의 직접 상호작용만 학습한다. "A가 본 영화 X를 좋아한 B가 좋아한 영화 Y"와 같은 **고차 연결 패턴**은 활용하지 못한다. 추천을 사용자-아이템 이분 그래프로 보면, GNN으로 이런 패턴을 포착할 수 있다.

**LightGCN의 핵심**: 추천에서는 복잡한 신경망이 오히려 방해된다. 사용자와 아이템은 ID만 가지고 있어 비선형 변환이 불필요하다. LightGCN은 가중치 행렬과 활성화 함수를 제거하고, **이웃 임베딩의 정규화된 합산**만 반복한다.

![사용자-아이템 이분 그래프](../diagram/ch12_recommender_bipartite.png)

![LightGCN 아키텍처](../diagram/ch12_lightgcn_architecture.png)

| 비교 항목 | 행렬 분해 (SVD) | LightGCN |
| --------- | --------------- | -------- |
| 정보 범위 | 직접 상호작용 (1-홉) | 다중 홉 (3-홉 이상) |
| 최적화 대상 | 평점 예측 (RMSE) | 랭킹 품질 (Recall, NDCG) |
| 변환 | 선형 분해 | 정규화된 이웃 집계 |

**BPR 손실 함수**: "본 영화 점수 > 안 본 영화 점수"가 되도록 상대 순서를 학습한다. 넷플릭스에서 중요한 것은 예측 평점 4.2 vs 4.3이 아니라, 좋아할 영화가 첫 화면에 나오느냐다.

**평가 지표**:

| 지표 | 측정하는 것 | 예시 해석 |
| ---- | ----------- | --------- |
| Recall@K | 정답 중 몇 개를 찾았나 | 0.1 = 정답의 10% 발견 |
| NDCG@K | 정답이 상위에 있나 | 0.12 = 상위 순위 정확도 |
| Hit Rate@K | 1개라도 맞혔나 | 0.7 = 70% 사용자 성공 |

---

### 🔬 실습 3: LightGCN 영화 추천

#### Step 1 — 실행

`practice/chapter12/code/12-4-lightgcn.py`를 실행한다.

```bash
python 12-4-lightgcn.py
```

출력에서 아래 표를 채운다.

**데이터셋 정보**:

| 항목 | 값 |
| ---- | -- |
| 사용자 수 | |
| 아이템 수 | |
| 평점 수 | |
| 희소성 | |

**LightGCN 추천 성능**:

| 지표 | 값 |
| ---- | -- |
| Recall@10 | |
| NDCG@10 | |

**11장 SVD와 비교**:

| 방법 | 평가 지표 | 값 | 비고 |
| ---- | --------- | -- | ---- |
| SVD (11장) | RMSE | | 평점 예측 |
| LightGCN | Recall@10 | | Top-N 추천 |
| LightGCN | NDCG@10 | | Top-N 추천 |

#### Step 2 — 이해

코드의 핵심 구조를 확인한다.

```python
# LightGCN: 활성화 함수와 가중치 변환 없이 이웃 집계만 수행
for _ in range(self.num_layers):
    out = torch.zeros_like(all_emb)
    out.index_add_(0, col, all_emb[row] * norm.unsqueeze(1))
    all_emb = out
    embs.append(all_emb)

# 모든 레이어 출력의 평균 (layer combination)
final_emb = torch.stack(embs, dim=0).mean(dim=0)
```

- `num_layers=3`이면 3-홉 이웃까지 정보가 전파된다. "A가 본 영화를 본 B가 본 영화를 본 C"의 정보까지 반영
- SVD와 LightGCN의 평가 지표가 다른 이유: SVD는 평점 예측 정확도(RMSE), LightGCN은 랭킹 품질(Recall, NDCG)을 최적화한다
- BPR 손실이 MSE가 아닌 이유: 추천에서는 절대 점수가 아닌 상대 순서가 중요하기 때문

#### Step 3 — 직접 코딩

**프롬프트 3**: LightGCN 레이어 수에 따른 성능 변화

> `12-4-lightgcn.py`를 수정해서, num_layers를 1, 2, 3, 4, 5로 바꾸며 Recall@10과 NDCG@10을 비교하는 표를 출력하는 코드를 작성해줘. epochs는 100으로 고정해줘.

결과를 기록한다:

| num_layers | Recall@10 | NDCG@10 |
| ---------- | --------- | ------- |
| 1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
| 5 | | |

- 레이어를 늘리면 성능이 계속 좋아지는가?
- LightGCN은 layer combination(모든 층의 평균)을 사용하므로 오버스무딩에 어느 정도 강건하다. GCN 노드 분류(실습 1)와 비교하면 어떤가?

**프롬프트 4** (선택): embedding_dim 변경 실험

> `12-4-lightgcn.py`를 수정해서, embedding_dim을 16, 32, 64, 128로 바꾸며 Recall@10과 NDCG@10을 비교하는 표를 출력하는 코드를 작성해줘.

---

### 12.5 지식 그래프 임베딩: 관계 추론

지식 그래프(KG)는 (머리 엔티티, 관계, 꼬리 엔티티) 형태의 트리플로 실세계 정보를 구조화한다. 예: (아인슈타인, 출생지, 독일). 지식 그래프 임베딩(KGE)은 엔티티와 관계를 벡터 공간에 매핑해 누락된 관계를 예측한다.

**TransE**: "머리 + 관계 ≈ 꼬리" (h + r ≈ t). 지도에서 "서울 + 동쪽 300km = 강릉"처럼 관계를 벡터 이동으로 모델링한다. 단순하지만 대칭 관계(결혼)나 1:N 관계(여러 국적)를 표현하지 못한다.

**RotatE**: 복소수 공간에서 관계를 **회전**으로 모델링한다 (t = h ∘ r). "결혼" 관계가 180도 회전이면 A→B, B→A 모두 성립한다.

![지식 그래프 개념](../diagram/ch12_knowledge_graph_concept.png)

![TransE 개념](../diagram/ch12_transe_concept.png)

| 관계 패턴 | 예시 | TransE | RotatE |
| --------- | ---- | ------ | ------ |
| 대칭 | A ↔ B (결혼) | X | O |
| 반대칭 | A → B (부모) | O | O |
| 역 | 부모 ↔ 자녀 | X | O |
| 합성 | 부 + 형 = 삼촌 | O | O |

---

### 🔬 실습 4: 지식 그래프 임베딩 — TransE vs RotatE

#### Step 1 — 실행

`practice/chapter12/code/12-5-kge.py`를 실행한다.

```bash
pip install pykeen  # 처음 한 번만
python 12-5-kge.py
```

출력에서 아래 표를 채운다.

**데이터셋 정보 (FB15k-237)**:

| 항목 | 값 |
| ---- | -- |
| 엔티티 수 | |
| 관계 수 | |
| 학습 트리플 | |
| 테스트 트리플 | |

**모델별 성능 비교**:

| 모델 | MRR | Hits@1 | Hits@3 | Hits@10 |
| ---- | --- | ------ | ------ | ------- |
| TransE | | | | |
| RotatE | | | | |

#### Step 2 — 이해

코드의 핵심 구조를 확인한다.

```python
# PyKEEN의 pipeline으로 간단하게 KGE 학습
result = pipeline(
    dataset=dataset,
    model='TransE',  # 또는 'RotatE'
    model_kwargs={'embedding_dim': 100},
    training_kwargs={'num_epochs': 100, 'batch_size': 256}
)
```

- RotatE가 TransE보다 모든 지표에서 우수한 이유: FB15k-237에 대칭·역 관계가 다수 포함되어 있어 회전 모델링이 유리
- MRR과 Hits@K의 차이: MRR은 정답의 평균 순위(전반적 랭킹 품질), Hits@K는 상위 K개에 정답이 포함된 비율
- Hits@1이 낮고 Hits@10이 높다면? 정답이 1위는 아니지만 상위 10개 안에는 들어온다는 뜻. 재순위화(re-ranking)로 개선 가능

#### Step 3 — 직접 코딩

**프롬프트 5**: embedding_dim 변경 실험

> `12-5-kge.py`를 수정해서, TransE의 embedding_dim을 50, 100, 200으로 바꾸며 MRR과 Hits@10을 비교하는 표를 출력하는 코드를 작성해줘. epochs는 100으로 고정해줘.

결과를 기록한다:

| embedding_dim | MRR | Hits@10 |
| ------------- | --- | ------- |
| 50 | | |
| 100 | | |
| 200 | | |

- 차원을 늘리면 성능이 계속 좋아지는가?
- 차원이 크면 표현력은 높아지지만 학습 데이터가 부족하면 과적합 위험이 있다

---

### 12.6 GraphRAG: 지식 그래프 + LLM

GraphRAG는 대규모 언어 모델(LLM)의 추론 능력과 지식 그래프의 구조화된 정보를 결합한다. 기존 Vector RAG는 질의와 유사한 문서를 검색하지만, GraphRAG는 엔티티 간 관계를 따라 다중 홉 추론이 가능하다.

| 특성 | Vector RAG | GraphRAG |
| ---- | ---------- | -------- |
| 인덱싱 | 문서 → 벡터 임베딩 | 문서 → 엔티티/관계 그래프 |
| 검색 | 유사도 기반 | 그래프 탐색 + 유사도 |
| 추론 | 단일 홉 | 다중 홉 |
| 전역 요약 | 어려움 | 커뮤니티 요약 활용 |
| 구축 비용 | 낮음 | 높음 (LLM 호출 필요) |
| 환각 감소 | 제한적 | 구조화된 근거 제공 |

---

### 🔬 실습 5: GraphRAG 개념 실습

#### Step 1 — 실행

`practice/chapter12/code/12-6-graphrag.py`를 실행한다.

```bash
python 12-6-graphrag.py
```

출력에서 아래 표를 채운다.

**지식 그래프 정보**:

| 항목 | 값 |
| ---- | -- |
| 엔티티 수 | |
| 관계 수 | |
| 엔티티 유형 수 | |

**Vector RAG vs GraphRAG 비교** (질의: "What is the connection between Einstein and quantum mechanics?"):

| 방법 | 검색 결과 요약 |
| ---- | -------------- |
| Vector RAG | |
| GraphRAG | |

#### Step 2 — 이해

코드의 핵심 구조를 확인한다.

```python
# 단일 홉 질의: 직접 연결된 엔티티 검색
for neighbor in G.neighbors(entity_id):
    edge_data = G.get_edge_data(entity_id, neighbor)
    # "Einstein → Theory of Relativity (developed)"

# 다중 홉 질의: DFS로 경로 탐색
# "Einstein → Bohr (debated_with) → Quantum Mechanics (contributed_to)"
```

- Vector RAG가 "Einstein과 quantum mechanics의 관계"를 잘 못 찾는 이유: 두 키워드가 같은 문서에 있어야 검색됨
- GraphRAG가 이를 찾을 수 있는 이유: Einstein → Bohr → Quantum Mechanics 경로를 그래프에서 탐색
- 커뮤니티 탐지가 전역 요약에 중요한 이유: "물리학자 그룹", "이론 그룹" 같은 구조적 요약 가능

#### Step 3 — 직접 코딩

**프롬프트 6**: 새로운 엔티티와 관계를 추가한 확장 실험

> `12-6-graphrag.py`의 SAMPLE_KNOWLEDGE_GRAPH에 다음 엔티티와 관계를 추가하고 실행해줘: (1) "Werner Heisenberg" (Person) - contributed_to → Quantum Mechanics, (2) "Uncertainty Principle" (Theory) - Heisenberg가 developed, (3) Heisenberg가 received → Nobel Prize. 추가 후 "Who contributed to Quantum Mechanics?"라는 질의에 대한 GraphRAG 결과를 출력해줘.

**프롬프트 7**: 지식 그래프 통계 분석

> `12-6-graphrag.py`의 지식 그래프에서 각 엔티티의 연결 수(차수)를 계산하고, 가장 연결이 많은 엔티티 Top-3를 출력하고, 엔티티 유형별 평균 차수를 비교하는 표를 출력하는 코드를 작성해줘.

결과를 기록한다:

**엔티티 유형별 평균 차수**:

| 엔티티 유형 | 평균 차수 |
| ----------- | --------- |
| Person | |
| Theory | |
| 기타 | |

- 가장 연결이 많은 엔티티는 지식 그래프에서 어떤 역할을 하는가?
- 이 분석이 GraphRAG의 질의 응답에 어떻게 도움이 되는가?

**프롬프트 8** (선택): Vector RAG 개선 — TF-IDF 유사도

> `12-6-graphrag.py`의 vector_rag_simulation 함수를 개선해서, 단순 키워드 매칭 대신 TF-IDF 코사인 유사도를 사용하는 버전을 작성해줘. sklearn의 TfidfVectorizer를 사용하고, 기존 키워드 매칭 결과와 비교해줘.

---

### 핵심 정리

```text
1. GNN은 메시지 패싱(생성→집계→갱신)으로 이웃 정보를 반복 집계한다. 2-4층이 실무 권장.
2. GCN은 단순 평균, GAT는 어텐션 가중치, GraphSAGE는 샘플링 기반. 규모와 요구에 따라 선택.
3. 노드 분류: GNN이 인용 관계를 통해 논문 주제를 자동 학습. 레이블 전파 효과.
4. 링크 예측: GNN이 전통적 방법(CN, Jaccard) 대비 노드 특성과 다중 홉 정보를 활용해 성능 향상.
5. LightGCN: 추천에서는 단순한 이웃 집계가 복잡한 신경망보다 효과적. BPR로 랭킹 학습.
6. KGE: TransE(이동)은 단순 관계에, RotatE(회전)은 대칭·역 관계에 강하다.
7. GraphRAG: Vector RAG의 단일 홉 한계를 그래프 탐색으로 극복. 다중 홉 추론과 전역 요약 가능.
```

---

### 제출 기준

- ✓ 5개 제공 코드를 모두 실행하고 결과 수치를 기록했다
- ✓ Step 3의 프롬프트 중 최소 4개 이상을 AI 도구로 코드를 작성하고 실행했다
- ✓ 직접 작성한 코드와 실행 결과를 포함했다
- ✓ 각 실습의 결과를 왜 그런 결과가 나왔는지 1~2문장으로 해석했다

### 바이브 코딩 팁

- AI 도구에 프롬프트를 줄 때, **데이터와 목표를 구체적으로** 적을수록 좋은 코드가 나온다
- 생성된 코드를 그대로 실행하지 말고, **코드를 읽고 이해한 뒤** 실행한다
- 에러가 나면 에러 메시지를 AI에 다시 붙여넣어 수정을 요청한다
- 결과가 예상과 다르면 왜 다른지 AI에게 물어본다
- **AI가 생성한 코드의 결과도 반드시 검증**한다 — 이것이 이론에서 배운 원칙의 실천이다
