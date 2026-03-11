# 3장 B: 분류와 회귀 — 모범 답안과 해설

> 이 문서는 실습 제출 후 공개한다. 제출 전에는 열람하지 않는다.

---

## 실습 1 해설: 타이타닉 생존 예측 의사결정나무

### 제공 코드 실행 결과 해설

타이타닉 데이터(835샘플, 4특성: Sex, 3rd_class, Age, 1st_class)에 의사결정나무(max_depth=4)를 적용한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 정확도 | 0.78~0.82 | 4개 특성만 사용한 단순 모델. 추가 특성(SibSp, Fare 등)이 있으면 더 올라갈 수 있음 |
| 특성 중요도 1위 | Sex (0.50 전후) | "여성과 아이를 먼저" 원칙이 실제 적용됨 |
| 특성 중요도 2위 | Age (0.20~0.25) | 남성은 37~38세, 여성은 13세(아동)가 분기점 |
| 특성 중요도 3위 | 3rd_class (0.15~0.20) | 구명보트 접근성의 계층적 차이 |
| 특성 중요도 4위 | 1st_class (0.05~0.10) | 3rd_class와 상관 → 추가 정보량 적음 |

핵심 코드 구조:

```python
# max_depth=4: 과적합 방지를 위한 사전 가지치기
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
dt_model.fit(X_train, y_train)

# 특성 중요도: 각 변수가 불순도 감소에 기여한 비율 (합 = 1)
feature_importance = dt_model.feature_importances_
```

트리의 첫 분할이 Sex인 이유: 성별로 나누면 지니 불순도가 가장 크게 감소한다. 남성 그룹은 사망 비율이 높고, 여성 그룹은 생존 비율이 높아 정보 이득이 최대다.

### 프롬프트 1 모범 구현: max_depth 변경 과적합 관찰

```python
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_path = Path(__file__).parent.parent / "data" / "titanic.csv"
df = pd.read_csv(data_path)

X = df[['Sex', '3rd_class', 'Age', '1st_class']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"{'max_depth':<12} {'훈련 정확도':<14} {'테스트 정확도':<14} {'차이':<10}")
print("-" * 50)

for depth in [2, 3, 4, 6, None]:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    gap = train_acc - test_acc
    label = str(depth) if depth else "None"
    print(f"{label:<12} {train_acc:<14.4f} {test_acc:<14.4f} {gap:<10.4f}")
```

기대 결과 해석:
- max_depth=2: 훈련/테스트 정확도가 비슷. 모델이 단순해서 과적합 없음(언더피팅 가능)
- max_depth=4: 훈련 정확도가 조금 높아지면서 테스트 정확도도 유지. 적절한 복잡도
- max_depth=None: 훈련 정확도가 거의 1.0에 가까워지지만 테스트 정확도는 떨어짐. **과적합**
- "훈련-테스트 정확도 차이(gap)"가 커지는 지점이 과적합의 시작이다

### 프롬프트 2 모범 구현: 혼동행렬 히트맵

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data_path = Path(__file__).parent.parent / "data" / "titanic.csv"
df = pd.read_csv(data_path)

X = df[['Sex', '3rd_class', 'Age', '1st_class']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
TP, FP, FN, TN = cm[1,1], cm[0,1], cm[1,0], cm[0,0]

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"정밀도: {precision:.4f}")
print(f"재현율: {recall:.4f}")
print(f"F1:     {f1:.4f}")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['사망', '생존'], yticklabels=['사망', '생존'])
plt.xlabel('예측')
plt.ylabel('실제')
plt.title('Titanic Decision Tree - Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix_heatmap.png", dpi=150)
plt.show()
```

기대 결과: 혼동행렬에서 FN(생존자를 사망으로 잘못 예측)이 FP보다 클 수 있다. 이는 모델이 보수적으로(사망 쪽으로) 예측하는 경향을 보여준다.

### 프롬프트 3 모범 구현: Random Forest와 비교

```python
import pandas as pd
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

data_path = Path(__file__).parent.parent / "data" / "titanic.csv"
df = pd.read_csv(data_path)

X = df[['Sex', '3rd_class', 'Age', '1st_class']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
}

print(f"{'모델':<20} {'정확도':<10} {'정밀도':<10} {'재현율':<10}")
print("-" * 50)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    print(f"{name:<20} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f}")
```

기대 결과:
- Random Forest가 단일 Decision Tree보다 정확도가 약간 높거나 비슷함
- 이 데이터는 특성이 4개뿐이라 앙상블 효과가 크지 않을 수 있음
- 특성이 더 많고 데이터가 복잡할수록 앙상블의 효과가 커진다
- 핵심: 여러 트리가 각각 다른 데이터 부분집합을 학습하고, 예측을 합치면 개별 트리의 오류가 상쇄된다

### 프롬프트 4 모범 구현: 교차검증

```python
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

data_path = Path(__file__).parent.parent / "data" / "titanic.csv"
df = pd.read_csv(data_path)

X = df[['Sex', '3rd_class', 'Age', '1st_class']]
y = df['Survived']

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
scores = cross_val_score(dt, X, y, cv=5, scoring='accuracy')

print("5-Fold 교차검증 결과:")
for i, s in enumerate(scores, 1):
    print(f"  Fold {i}: {s:.4f}")
print(f"\n평균 정확도: {scores.mean():.4f} ± {scores.std():.4f}")
```

핵심: 교차검증은 데이터를 5등분하여 매번 다른 1/5를 테스트셋으로 사용한다. 표준편차가 작으면 모델이 안정적이고, 크면 데이터 분할에 민감하다는 뜻이다.

---

## 실습 2 해설: 이상치 탐지 알고리즘 비교

### 제공 코드 실행 결과 해설

합성 신용카드 거래 데이터(10,050건, 사기 약 0.5%)에 4가지 알고리즘을 적용한 결과:

| 알고리즘 | F1 경향 | 이유 |
| -------- | ------- | ---- |
| Isolation Forest | 가장 높음 (0.40~0.44) | 이상치의 빠른 고립 특성을 효과적으로 포착 |
| LOF | 두 번째 (0.35~0.38) | 밀도 기반으로 안정적. 국소 이상치 탐지에 강함 |
| One-class SVM | 낮음 (0.10~0.15) | 5차원 단순 데이터에서 커널 경계가 비효율적 |
| Autoencoder | 낮음 (0.10) | 단순 합성 데이터에서는 과한 모델. 고차원 실제 데이터에서 강점 |

핵심 코드 구조:

```python
# contamination: "전체 데이터 중 이상치 비율이 이 정도"라고 알려주는 파라미터
contamination = n_fraud / len(y)  # 실제 사기 비율 사용

# Isolation Forest: 원본 데이터 사용
iso_forest = IsolationForest(contamination=contamination, random_state=42)
y_pred_iso = iso_forest.fit_predict(X)

# One-class SVM, Autoencoder: 스케일링된 데이터 사용
X_scaled = scaler.fit_transform(X)
ocsvm = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
y_pred_svm = ocsvm.fit_predict(X_scaled)
```

Isolation Forest는 스케일링 없이도 작동하는 이유: 무작위 분할 기반이라 변수 스케일에 영향받지 않는다. 반면 One-class SVM과 Autoencoder는 거리/기울기 기반이라 스케일링이 필요하다.

### 프롬프트 5 모범 구현: contamination 비율 변경

```python
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score

data_path = Path(__file__).parent.parent / "data" / "credit_card_fraud.csv"
df = pd.read_csv(data_path)

feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
X = df[feature_cols].values
y = df['label'].values

contaminations = [0.005, 0.01, 0.02, 0.05]

print(f"{'contamination':<16} {'정밀도':<10} {'재현율':<10} {'F1':<10}")
print("-" * 46)

for cont in contaminations:
    iso = IsolationForest(contamination=cont, random_state=42, n_jobs=-1)
    y_pred = iso.fit_predict(X)

    y_true_bin = (y == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)

    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    print(f"{cont:<16} {prec:<10.3f} {rec:<10.3f} {f1:<10.3f}")
```

기대 결과 해석:
- contamination을 높이면: 더 많은 거래를 이상치로 판단 → 재현율은 올라가지만 정밀도는 내려감
- contamination을 낮추면: 적게 잡으므로 정밀도는 올라가지만 놓치는 사기가 많아짐
- 실무에서는 후속 검토 인력/시스템 용량과 사기 한 건당 손실을 고려해 결정한다
- 예: 검토 인력이 하루 100건 처리 가능하면, 일일 거래 10,000건 중 상위 1%(contamination=0.01)를 검토 대상으로 설정

### 프롬프트 6 모범 구현: anomaly score 분포 시각화

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.ensemble import IsolationForest

data_path = Path(__file__).parent.parent / "data" / "credit_card_fraud.csv"
df = pd.read_csv(data_path)

feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
X = df[feature_cols].values
y = df['label'].values

iso = IsolationForest(contamination=0.005, random_state=42, n_jobs=-1)
iso.fit(X)
scores = iso.decision_function(X)  # 이상치 점수 (낮을수록 이상)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

axes[0].hist(scores[y == 1], bins=50, color='steelblue', alpha=0.7)
axes[0].set_title('Normal Transactions')
axes[0].set_xlabel('Anomaly Score')
axes[0].set_ylabel('Count')

axes[1].hist(scores[y == -1], bins=20, color='crimson', alpha=0.7)
axes[1].set_title('Fraud Transactions')
axes[1].set_xlabel('Anomaly Score')

plt.suptitle('Isolation Forest Anomaly Score Distribution', fontsize=14)
plt.tight_layout()
plt.savefig("anomaly_score_distribution.png", dpi=150)
plt.show()

print(f"정상 거래 점수: 평균={scores[y==1].mean():.3f}, 표준편차={scores[y==1].std():.3f}")
print(f"사기 거래 점수: 평균={scores[y==-1].mean():.3f}, 표준편차={scores[y==-1].std():.3f}")
```

기대 결과:
- 정상 거래: 이상치 점수가 대부분 0 이상(양수)에 분포
- 사기 거래: 이상치 점수가 0 근처이거나 음수 쪽으로 치우침
- 두 분포가 잘 분리되면 Isolation Forest가 효과적이라는 뜻
- `decision_function`은 양수가 정상, 음수가 이상치를 의미한다

### 프롬프트 7 모범 구현: 알고리즘별 성능 막대 그래프

```python
import numpy as np
import matplotlib.pyplot as plt

# 실행 결과에서 가져온 값 (실제 실행 결과로 대체)
algorithms = ['Isolation\nForest', 'One-class\nSVM', 'LOF', 'Autoencoder']
precision_vals = [0.440, 0.120, 0.380, 0.100]
recall_vals = [0.440, 0.200, 0.380, 0.100]
f1_vals = [0.440, 0.150, 0.380, 0.100]

x = np.arange(len(algorithms))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, precision_vals, width, label='Precision', color='steelblue')
bars2 = ax.bar(x, recall_vals, width, label='Recall', color='coral')
bars3 = ax.bar(x + width, f1_vals, width, label='F1', color='seagreen')

ax.set_xlabel('Algorithm')
ax.set_ylabel('Score')
ax.set_title('Anomaly Detection: Algorithm Comparison')
ax.set_xticks(x)
ax.set_xticklabels(algorithms)
ax.legend()
ax.set_ylim(0, 0.6)

# 값 표시
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("algorithm_comparison.png", dpi=150)
plt.show()
```

기대 결과:
- Isolation Forest와 LOF가 세 지표 모두 높은 균형 잡힌 성능
- One-class SVM은 재현율이 정밀도보다 상대적으로 높음 (많이 잡지만 오탐도 많음)
- 재현율만 높은 것이 항상 좋지 않은 이유: FP가 많으면 검토 비용이 폭증하고, 실무 팀이 "늑대와 소년" 효과로 경고를 무시하게 됨

### 프롬프트 8 모범 구현: Autoencoder 구조 변경

```python
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

data_path = Path(__file__).parent.parent / "data" / "credit_card_fraud.csv"
df = pd.read_csv(data_path)

feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
X = df[feature_cols].values
y = df['label'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_normal = X_scaled[y == 1]
contamination = (y == -1).sum() / len(y)

torch.manual_seed(42)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))

hidden_dims = [4, 8, 16, 32]
print(f"{'hidden_dim':<14} {'F1':<10} {'시간':<10}")
print("-" * 34)

for h_dim in hidden_dims:
    torch.manual_seed(42)
    model = Autoencoder(X_scaled.shape[1], h_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    X_tensor = torch.FloatTensor(X_normal)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    start = time.time()
    model.train()
    for _ in range(50):
        for batch_x, _ in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_x)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_all = torch.FloatTensor(X_scaled)
        recon = model(X_all).numpy()
    mse = np.mean((X_scaled - recon) ** 2, axis=1)
    threshold = np.percentile(mse, (1 - contamination) * 100)
    y_pred = np.where(mse > threshold, -1, 1)
    elapsed = time.time() - start

    y_true_bin = (y == -1).astype(int)
    y_pred_bin = (y_pred == -1).astype(int)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    print(f"{h_dim:<14} {f1:<10.3f} {elapsed:<10.2f}초")
```

기대 결과:
- hidden_dim=4: 가장 작은 구조. 압축이 심해 정상/이상 구분이 어려울 수 있음
- hidden_dim=8: 기본값. 적절한 압축 수준
- hidden_dim=16, 32: 구조가 커지면 모든 데이터를 잘 복원 → 이상치도 잘 복원되어 탐지력 저하 가능
- 핵심: Autoencoder는 "적당히 좁은 병목"이 있어야 이상치를 잘 거른다. 너무 크면 이상치도 잘 통과한다

---

## 3장 전체 핵심 정리

```text
1. 의사결정나무: max_depth가 커지면 훈련 정확도는 올라가지만 테스트 정확도는 떨어진다.
   이 격차(gap)가 과적합의 신호다.
2. 혼동행렬: TP/FP/FN/TN에서 정밀도/재현율/F1을 직접 계산할 수 있어야 한다.
   비즈니스 맥락에서 FP와 FN 중 어느 비용이 더 큰지가 지표 선택을 결정한다.
3. 앙상블(Random Forest): 단일 트리의 과적합을 여러 트리의 평균으로 완화한다.
   데이터가 복잡할수록 앙상블 효과가 커진다.
4. 이상치 탐지: contamination 파라미터가 정밀도/재현율 트레이드오프를 직접 조절한다.
   실무에서는 후속 검토 용량과 놓침 비용을 함께 고려한다.
5. Isolation Forest가 단순 데이터에서 가장 효과적이지만, 고차원 실제 데이터에서는
   Autoencoder가 강점을 보일 수 있다. 알고리즘 선택은 데이터 특성에 따라 달라진다.
6. AI 도구로 코드를 생성하되, 결과를 반드시 검증하고 해석하는 습관이 중요하다.
```
