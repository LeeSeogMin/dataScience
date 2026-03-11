# 7장 B: 신경망 기반 데이터 분석 — 모범 답안과 해설

> 이 문서는 실습 제출 후 공개한다. 제출 전에는 열람하지 않는다.

---

## 실습 1 해설: 정형 데이터 모델 비교 (Adult Income)

### 제공 코드 실행 결과 해설

Adult Income 데이터셋(약 45,000건, 14특성)에 4개 모델을 적용한 결과:

| 모델 | 정확도 경향 | AUC 경향 | 학습 시간 경향 | 이유 |
| ---- | ----------- | -------- | -------------- | ---- |
| XGBoost | 0.865 | 0.925 | ~0.5초 | 트리 기반 모델이 단순 정형 데이터에서 최적 |
| MLP | 0.847 | 0.905 | ~25초 | 기본 구조, 특성 선택 없음 |
| TabNet | 0.833 | 0.890 | ~160초 | 데이터 규모 대비 모델이 과함 |
| FT-Transformer | 0.805 | 0.842 | ~880초 | 14특성에서 Attention 오버헤드 큼 |

핵심 코드 구조:

```python
# XGBoost: early_stopping_rounds로 자동 조기 중단
xgb = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                     eval_metric='auc', early_stopping_rounds=20)

# SimpleMLP: BatchNorm + Dropout으로 정규화
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.2):
        # Linear → ReLU → BatchNorm → Dropout 반복

# TabNet: 핵심 하이퍼파라미터
tabnet = TabNetClassifier(n_d=32, n_a=32, n_steps=5, gamma=1.5)
```

XGBoost가 최고 성능을 보이는 이유: Adult 데이터셋은 14개 특성에 상호작용이 단순하여, 트리 기반 모델의 축 정렬 분할이 효율적이다. 딥러닝 모델은 이 규모에서 과한 용량을 가지며, 학습 시간 대비 성능 이득이 없다.

### 프롬프트 1 모범 구현: XGBoost 하이퍼파라미터 튜닝

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from itertools import product

# 데이터 로드 (7-3 코드의 load_adult_data 함수와 동일)
adult = fetch_openml(name='adult', version=2, as_frame=True)
df = adult.frame
df['target'] = (df['class'] == '>50K').astype(int)
df = df.replace('?', np.nan).dropna()

X = df.drop(['class', 'target'], axis=1)
y = df['target'].values

cat_cols = X.select_dtypes(include=['object', 'category']).columns
for col in cat_cols:
    X[col] = LabelEncoder().fit_transform(X[col].astype(str))

X = X.values.astype(np.float32)
y = y.astype(np.int64)

# 분할
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# 그리드 서치
max_depths = [3, 4, 6, 8, 10]
n_estimators_list = [100, 200, 300]

results = []
for depth, n_est in product(max_depths, n_estimators_list):
    xgb = XGBClassifier(
        max_depth=depth, n_estimators=n_est, learning_rate=0.1,
        random_state=42, eval_metric='auc', early_stopping_rounds=20, verbosity=0
    )
    xgb.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

    val_prob = xgb.predict_proba(X_valid)[:, 1]
    test_prob = xgb.predict_proba(X_test)[:, 1]

    results.append({
        'max_depth': depth, 'n_estimators': n_est,
        'val_auc': roc_auc_score(y_valid, val_prob),
        'test_auc': roc_auc_score(y_test, test_prob)
    })

results_df = pd.DataFrame(results).sort_values('val_auc', ascending=False)
print(results_df.head(10).to_string(index=False))

best = results_df.iloc[0]
print(f"\n최적: max_depth={int(best['max_depth'])}, n_estimators={int(best['n_estimators'])}")
print(f"검증 AUC: {best['val_auc']:.4f}, 테스트 AUC: {best['test_auc']:.4f}")
```

기대 결과 해석:
- 기본 설정(max_depth=6, n_estimators=200)과 최적 설정의 AUC 차이는 0.001~0.005 수준으로 미미함
- 이는 XGBoost가 기본 설정으로도 안정적 성능을 보임을 입증
- "정형 데이터는 XGBoost 먼저" 원칙이 타당한 근거

### 프롬프트 2 모범 구현: TabNet 특성 중요도 시각화

```python
import matplotlib.pyplot as plt
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier

# TabNet 학습 후 (7-3 코드 실행 후)
# tabnet.feature_importances_ 로 특성 중요도 접근

feature_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num',
                 'marital-status', 'occupation', 'relationship', 'race', 'sex',
                 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

importances = tabnet.feature_importances_
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(12, 6))
plt.bar(range(len(importances)), importances[sorted_idx], color='steelblue')
plt.xticks(range(len(importances)),
           [feature_names[i] for i in sorted_idx], rotation=45, ha='right')
plt.ylabel('Feature Importance')
plt.title('TabNet Feature Importances (Adult Income)')
plt.tight_layout()
plt.savefig("tabnet_feature_importance.png", dpi=150)
plt.show()
```

기대 결과:
- TabNet은 Attention 기반으로 명시적 특성 선택을 수행하므로, 일부 특성에 가중치가 집중됨
- relationship, marital-status, capital-gain 등이 높은 중요도를 보일 가능성이 높음
- XGBoost의 feature_importances_와 비교하면, 선택하는 특성은 비슷하지만 분포가 더 희소(sparse)함

---

## 실습 2 해설: 합성 데이터에서 딥러닝의 강점 확인

### 제공 코드 실행 결과 해설

고차원 합성 데이터(50,000건, 100특성, 30개 정보성, 20개 비선형 상호작용)에서의 비교:

| 모델 | AUC 경향 | 이유 |
| ---- | -------- | ---- |
| XGBoost | 0.976 | 축 정렬 분할이 복잡한 상호작용에서는 비효율적 |
| MLP | 0.980 | 비선형 활성화가 상호작용 포착에 유리 |
| TabNet | 0.982 | Attention이 중요 상호작용을 명시적으로 선택 |
| FT-Transformer | ~0.55 | 기본 설정에서 과적합, 범주형 특성이 없어 불리 |

Adult와 결과가 역전된 핵심 이유: 곱셈, 삼각함수, 임계값 기반 상호작용은 축 정렬 분할로 포착하기 어렵다. 신경망은 이런 복잡한 비선형 관계를 연속적 변환으로 학습한다.

### 프롬프트 3 모범 구현: 데이터 크기에 따른 모델 성능 변화

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 7-3b의 generate_complex_synthetic_data 함수 import 또는 복사
# from ... import generate_complex_synthetic_data

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim), nn.ReLU(),
                nn.BatchNorm1d(h_dim), nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

sample_sizes = [1000, 5000, 10000, 30000, 50000]
results = []

for n in sample_sizes:
    X, y, _, _ = generate_complex_synthetic_data(
        n_samples=n, n_features=100, n_informative=30, n_interactions=20
    )

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)
    X_test_s = scaler.transform(X_test)

    # XGBoost
    xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42,
                         eval_metric='auc', early_stopping_rounds=20, verbosity=0)
    xgb.fit(X_train_s, y_train, eval_set=[(X_valid_s, y_valid)], verbose=False)
    xgb_auc = roc_auc_score(y_test, xgb.predict_proba(X_test_s)[:, 1])

    # MLP
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_s), torch.LongTensor(y_train)),
        batch_size=256, shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_valid_s), torch.LongTensor(y_valid)),
        batch_size=256
    )

    mlp = SimpleMLP(input_dim=100)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    patience = 0
    for epoch in range(100):
        mlp.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(mlp(xb), yb.float())
            loss.backward()
            optimizer.step()

        mlp.eval()
        with torch.no_grad():
            val_pred = torch.sigmoid(mlp(torch.FloatTensor(X_valid_s))).numpy()
        val_auc = roc_auc_score(y_valid, val_pred)
        if val_auc > best_auc:
            best_auc = val_auc
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break

    mlp.eval()
    with torch.no_grad():
        test_pred = torch.sigmoid(mlp(torch.FloatTensor(X_test_s))).numpy()
    mlp_auc = roc_auc_score(y_test, test_pred)

    results.append({'n_samples': n, 'XGBoost_AUC': xgb_auc, 'MLP_AUC': mlp_auc})
    print(f"n={n:>6}: XGBoost={xgb_auc:.4f}, MLP={mlp_auc:.4f}")

import pandas as pd
print("\n" + pd.DataFrame(results).to_string(index=False))
```

기대 결과 해석:
- n=1,000: XGBoost가 MLP보다 높음. 데이터가 적으면 트리 모델이 안정적
- n=5,000~10,000: 비슷한 성능. 딥러닝이 따라잡기 시작
- n=30,000~50,000: MLP가 XGBoost를 앞섬. 충분한 데이터에서 비선형 표현력이 효과 발휘
- 핵심: 딥러닝은 데이터가 충분해야(보통 1만 건 이상) 트리 모델을 앞서기 시작한다

---

## 실습 3 해설: 오토인코더 + XGBoost 하이브리드

### 제공 코드 실행 결과 해설

합성 데이터(50,000건, 100특성)에서 오토인코더 잠재 표현의 활용:

| 특성 세트 | 특성 수 | AUC 경향 | 이유 |
| --------- | ------- | -------- | ---- |
| 원본 특성 | 100 | 0.975~0.980 | 기본 성능 |
| 잠재 표현 | 32 | 0.960~0.975 | 압축 과정에서 일부 정보 손실 |
| 원본 + 잠재 | 132 | 0.978~0.985 | 비선형 관계가 새로운 특성으로 추가 |

핵심 코드 구조:

```python
# 오토인코더: 입력 100차원 → 잠재 32차원
autoencoder = Autoencoder(input_dim=X_train_scaled.shape[1], latent_dim=32)

# 잠재 표현 추출
latent_train = extract_latent(autoencoder, X_train_scaled, device)

# 3가지 조합으로 XGBoost 학습
combos = [
    ("원본 특성", X_train_scaled, ...),
    ("잠재 표현", latent_train, ...),
    ("원본 + 잠재", np.hstack([X_train_scaled, latent_train]), ...)
]
```

원본 + 잠재 결합이 효과적인 이유: 오토인코더의 잠재 표현은 특성 간 비선형 관계를 압축한 것이다. 이 압축된 표현이 트리 모델에 "미리 계산된 특성 상호작용"으로 제공되어, XGBoost가 더 쉽게 패턴을 학습할 수 있다.

### 프롬프트 4 모범 구현: 잠재 차원 변경 실험

```python
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

# 데이터 로드
data_dir = Path(__file__).resolve().parent.parent / "data"
df = pd.read_csv(data_dir / "synthetic_complex.csv")
X = df.drop('target', axis=1).values.astype(np.float32)
y = df['target'].values.astype(np.int64)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_valid_s = scaler.transform(X_valid)
X_test_s = scaler.transform(X_test)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, input_dim)
        )
    def forward(self, x):
        return self.decoder(self.encoder(x))
    def get_latent(self, x):
        return self.encoder(x)

latent_dims = [8, 16, 32, 64, 128]
results = []

for l_dim in latent_dims:
    torch.manual_seed(42)
    start = time.time()

    ae = Autoencoder(X_train_s.shape[1], l_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train_s)),
        batch_size=512, shuffle=True
    )

    for epoch in range(60):
        ae.train()
        for batch, in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = criterion(ae(batch), batch)
            loss.backward()
            optimizer.step()

    ae.eval()
    with torch.no_grad():
        z_train = ae.get_latent(torch.FloatTensor(X_train_s).to(device)).cpu().numpy()
        z_valid = ae.get_latent(torch.FloatTensor(X_valid_s).to(device)).cpu().numpy()
        z_test = ae.get_latent(torch.FloatTensor(X_test_s).to(device)).cpu().numpy()

    # 원본 + 잠재 결합
    X_tr_comb = np.hstack([X_train_s, z_train])
    X_val_comb = np.hstack([X_valid_s, z_valid])
    X_te_comb = np.hstack([X_test_s, z_test])

    xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42,
                         eval_metric='auc', early_stopping_rounds=20, verbosity=0)
    xgb.fit(X_tr_comb, y_train, eval_set=[(X_val_comb, y_valid)], verbose=False)
    auc = roc_auc_score(y_test, xgb.predict_proba(X_te_comb)[:, 1])
    elapsed = time.time() - start

    results.append({'latent_dim': l_dim, 'AUC': auc, 'time': elapsed})
    print(f"latent_dim={l_dim:>4}: AUC={auc:.4f}, 시간={elapsed:.1f}초")

print("\n" + pd.DataFrame(results).to_string(index=False))
```

기대 결과 해석:
- latent_dim=8: 압축이 너무 심해 정보 손실 → AUC 상대적으로 낮음
- latent_dim=32~64: 적절한 균형. 핵심 정보 보존하면서 효과적 압축
- latent_dim=128: 입력 차원(100)보다 큰 잠재 차원. 압축 효과 감소, 오히려 노이즈가 포함될 수 있음
- 핵심: "적당히 좁은 병목"이 있어야 핵심 정보만 남는다

### 프롬프트 5 모범 구현: 잠재 표현 t-SNE 시각화

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# 오토인코더 학습 후 잠재 표현 추출 (위 코드에서 z_test 사용)
# 1000건만 샘플링
np.random.seed(42)
sample_idx = np.random.choice(len(z_test), 1000, replace=False)
z_sample = z_test[sample_idx]
y_sample = y_test[sample_idx]

# t-SNE 적용
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
z_2d = tsne.fit_transform(z_sample)

# 시각화
plt.figure(figsize=(8, 6))
for label, color, name in [(0, 'steelblue', 'Class 0'), (1, 'coral', 'Class 1')]:
    mask = y_sample == label
    plt.scatter(z_2d[mask, 0], z_2d[mask, 1], c=color, label=name, alpha=0.5, s=10)

plt.legend()
plt.title('Autoencoder Latent Representation (t-SNE)')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.tight_layout()
plt.savefig("latent_tsne.png", dpi=150)
plt.show()
```

기대 결과:
- 잘 학습된 경우: 두 클래스가 어느 정도 분리된 군집을 형성
- 분리가 명확할수록 잠재 표현이 분류에 유용한 정보를 포착했다는 의미
- 완전히 분리되지 않는 것이 정상 — 오토인코더는 비지도 학습으로 클래스 정보를 직접 사용하지 않았기 때문

---

## 실습 4 해설: 고차원 중복 특성에서 오토인코더의 가치

### 제공 코드 실행 결과 해설

1000차원 중복 데이터(핵심 100개 + 파생 900개)에서의 비교:

| 특성 세트 | 특성 수 | AUC 경향 | 이유 |
| --------- | ------- | -------- | ---- |
| 전체 (1000개) | 1000 | 0.80~0.83 | 중복 특성이 노이즈로 작용 |
| 핵심만 (Oracle) | 100 | 0.83~0.86 | 노이즈 없는 최적 조건 (실무에서 알 수 없음) |
| 잠재 표현 (64개) | 64 | 0.79~0.82 | 중복 제거된 압축 표현 |
| 전체 + 잠재 | 1064 | 0.82~0.85 | 압축 표현이 추가 신호 제공 |

핵심 코드 구조:

```python
# 핵심 100개 + 파생(중복) 900개 = 1000개
# 파생 특성: 2~5개 핵심 특성의 선형 조합 + 노이즈
X, y, n_core = generate_redundant_data(
    n_samples=50000, n_core=100, n_derived=900, noise_level=0.3
)

# 1000차원 → 64차원으로 압축 (15.6배 압축)
ae = RedundantAutoencoder(input_dim=X_train.shape[1], latent_dim=64)
```

핵심 인사이트:
- 압축률 15.6배(1000→64)에도 불구하고 전체 특성 대비 95% 이상 성능 유지
- "핵심만 (Oracle)"에 가장 근접한 것은 "전체 + 잠재" 조합
- 실무에서는 어떤 특성이 핵심인지 모르므로, 오토인코더가 자동으로 중복을 제거하는 역할

### 프롬프트 6 모범 구현: PCA와 오토인코더 비교

```python
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# generate_redundant_data 함수 (7-4b-ae-xgboost-redundant.py에서 복사)
# X, y, n_core = generate_redundant_data(...)

# 데이터 분할 및 스케일링
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype(np.float32)
X_test_s = scaler.transform(X_test).astype(np.float32)

# 1) 원본 특성 XGBoost
xgb_orig = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, verbosity=0)
xgb_orig.fit(X_train_s, y_train)
auc_orig = roc_auc_score(y_test, xgb_orig.predict_proba(X_test_s)[:, 1])

# 2) PCA (64차원)
pca = PCA(n_components=64, random_state=42)
X_train_pca = pca.fit_transform(X_train_s)
X_test_pca = pca.transform(X_test_s)

xgb_pca = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, verbosity=0)
xgb_pca.fit(X_train_pca, y_train)
auc_pca = roc_auc_score(y_test, xgb_pca.predict_proba(X_test_pca)[:, 1])

# 3) AE 잠재 표현 (64차원)
class RedundantAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
            nn.Linear(512, input_dim)
        )
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
ae = RedundantAutoencoder(X_train_s.shape[1], 64).to(device)
optimizer = torch.optim.AdamW(ae.parameters(), lr=1e-3, weight_decay=1e-5)
criterion = nn.MSELoss()
loader = DataLoader(TensorDataset(torch.FloatTensor(X_train_s)), batch_size=256, shuffle=True)

for epoch in range(60):
    ae.train()
    for batch, in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        recon, _ = ae(batch)
        loss = criterion(recon, batch)
        loss.backward()
        optimizer.step()

ae.eval()
with torch.no_grad():
    _, z_train = ae(torch.FloatTensor(X_train_s).to(device))
    _, z_test = ae(torch.FloatTensor(X_test_s).to(device))
    z_train = z_train.cpu().numpy()
    z_test = z_test.cpu().numpy()

xgb_ae = XGBClassifier(n_estimators=200, max_depth=6, random_state=42, verbosity=0)
xgb_ae.fit(z_train, y_train)
auc_ae = roc_auc_score(y_test, xgb_ae.predict_proba(z_test)[:, 1])

print(f"{'특성 세트':<22} | {'특성 수':>6} | {'AUC':>8}")
print("-" * 45)
print(f"{'원본 (1000개)':<22} | {1000:>6} | {auc_orig:>8.4f}")
print(f"{'PCA (64차원)':<22} | {64:>6} | {auc_pca:>8.4f}")
print(f"{'AE 잠재 표현 (64차원)':<22} | {64:>6} | {auc_ae:>8.4f}")
```

기대 결과 해석:
- PCA(64차원): 선형 변환으로 분산 기준 상위 성분 보존. 성능 양호
- AE 잠재 표현(64차원): 비선형 변환으로 더 복잡한 관계 포착
- 이 데이터에서는 파생 특성이 선형 조합이므로 PCA도 효과적. 비선형 관계가 강한 데이터에서는 AE가 더 유리
- 핵심: PCA는 선형, AE는 비선형. 데이터의 구조에 따라 선택

---

## 실습 5 해설: PyTorch 실무 파이프라인 구현

### 제공 코드 실행 결과 해설

합성 데이터(50,000건, 100특성)에 정규화가 적용된 MLP 파이프라인:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 학습 완료 에폭 | 40~50 (조기 중단) | patience=15로 충분한 대기 후 중단 |
| 최적 검증 손실 | 0.18~0.20 | 정규화로 안정적 학습 |
| 테스트 AUC | 0.978~0.982 | Dropout + Weight Decay + 조기 중단 효과 |

핵심 코드 구조:

```python
# 정규화 3중 적용
# 1. BatchNorm: 층 입력 정규화
# 2. Dropout(0.3): 뉴런 30% 무작위 비활성화
# 3. Weight Decay(1e-5): 가중치 크기 페널티

model = MLP(input_dim=100, hidden_dims=[128, 64, 32], dropout=0.3)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# 조기 중단: patience=15
early_stopping = EarlyStopping(patience=15)

# 학습률 스케줄러: 검증 손실 정체 시 학습률 절반
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
```

`model.train()` vs `model.eval()`의 차이:
- train(): Dropout이 활성화되어 뉴런을 무작위로 끔. BatchNorm은 현재 배치의 통계 사용
- eval(): Dropout 비활성화(모든 뉴런 사용). BatchNorm은 학습 중 누적된 전체 통계 사용

### 프롬프트 7 모범 구현: Dropout 비율 변경 실험

```python
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import os

# 데이터 로드 (7-5 코드와 동일)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(script_dir), 'data', 'synthetic_complex.csv')
df = pd.read_csv(data_path)
X = df.drop('target', axis=1).values.astype(np.float32)
y = df['target'].values.astype(np.int64)

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

class MLP(nn.Module):
    def __init__(self, input_dim, dropout=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.network(x).squeeze(-1)

dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]
results = []

for dr in dropout_rates:
    # 재현성
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model = MLP(X_train.shape[1], dropout=dr)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
        batch_size=256, shuffle=True, drop_last=True
    )

    best_loss = float('inf')
    patience = 0
    epochs_done = 0

    for epoch in range(100):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb.float())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out = model(torch.FloatTensor(X_valid))
            val_loss = criterion(val_out, torch.LongTensor(y_valid).float()).item()

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 15:
                break

        epochs_done = epoch + 1

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = torch.sigmoid(model(torch.FloatTensor(X_test))).numpy()
    test_auc = roc_auc_score(y_test, test_pred)

    results.append({'dropout': dr, 'epochs': epochs_done, 'test_auc': test_auc})
    print(f"Dropout={dr:.1f}: epochs={epochs_done}, AUC={test_auc:.4f}")

print("\n" + pd.DataFrame(results).to_string(index=False))
```

기대 결과 해석:
- Dropout=0.0: 정규화 없이 학습 → 과적합 위험. 빠르게 학습하지만 테스트 AUC가 상대적으로 낮을 수 있음
- Dropout=0.1~0.3: 적절한 정규화 → 최적 성능. 학습이 조금 느리지만 일반화 우수
- Dropout=0.5: 과도한 정규화 → 언더피팅. 뉴런의 절반이 꺼지면 학습 자체가 어려워짐
- 핵심: Dropout은 "적당히"가 중요. 일반적으로 0.1~0.3이 효과적

### 프롬프트 8 모범 구현: 학습 곡선 시각화

```python
import matplotlib.pyplot as plt
import numpy as np

# 7-5 코드 실행 후 history 딕셔너리 사용
# history = {'train_loss': [...], 'valid_loss': [...], 'valid_auc': [...]}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

epochs = range(1, len(history['train_loss']) + 1)

# 왼쪽: 손실 곡선
ax1.plot(epochs, history['train_loss'], label='Train Loss', color='steelblue')
ax1.plot(epochs, history['valid_loss'], label='Valid Loss', color='coral')
ax1.axvline(x=len(epochs), color='gray', linestyle='--', alpha=0.5, label='Early Stop')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 오른쪽: AUC 곡선
ax2.plot(epochs, history['valid_auc'], label='Valid AUC', color='seagreen')
ax2.axvline(x=len(epochs), color='gray', linestyle='--', alpha=0.5, label='Early Stop')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('AUC')
ax2.set_title('Validation AUC')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('PyTorch Pipeline - Learning Curves', fontsize=14)
plt.tight_layout()
plt.savefig("learning_curves.png", dpi=150)
plt.show()

print(f"학습 에폭: {len(epochs)}")
print(f"최종 Train Loss: {history['train_loss'][-1]:.4f}")
print(f"최저 Valid Loss: {min(history['valid_loss']):.4f} (Epoch {np.argmin(history['valid_loss'])+1})")
print(f"최고 Valid AUC: {max(history['valid_auc']):.4f} (Epoch {np.argmax(history['valid_auc'])+1})")
```

기대 결과:
- Train Loss: 지속적으로 하락
- Valid Loss: 초반 하락 후 일정 시점부터 정체 또는 상승 → 이 지점이 과적합 시작
- Valid AUC: 초반 급격히 상승 후 안정화
- 조기 중단 시점(세로 점선)이 Valid Loss가 정체되기 시작한 직후에 위치
- 핵심: Train Loss와 Valid Loss의 간격이 과적합의 정도를 보여준다

---

## 7장 전체 핵심 정리

```text
1. 정형 데이터는 XGBoost 먼저: Adult 데이터(14특성, 단순)에서 XGBoost가 AUC 0.925로 최고.
   딥러닝은 학습 시간만 길고 성능 이점이 없다.
2. 딥러닝이 유리한 조건: 고차원(100특성)과 복잡한 상호작용이 있으면 TabNet이 AUC 0.982로 역전.
   데이터가 1만 건 이상이어야 딥러닝의 장점이 드러난다.
3. 하이브리드 전략: 오토인코더 잠재 표현 + XGBoost 조합이 원본만 사용할 때보다 성능이 좋다.
   잠재 표현이 "미리 계산된 특성 상호작용"을 제공하기 때문이다.
4. 중복 특성 제거: 1000차원을 64차원으로 15.6배 압축해도 95% 이상 성능 유지.
   오토인코더가 핵심 정보만 자동으로 추출한다.
5. 정규화의 중요성: Dropout 0.1~0.3, Weight Decay, 조기 중단을 함께 적용해야
   과적합을 방지하고 일반화 성능을 확보할 수 있다.
6. AI 도구로 코드를 생성하되, 결과를 반드시 검증하고 해석하는 습관이 중요하다.
```
