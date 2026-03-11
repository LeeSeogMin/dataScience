# 10장 B: 생존 분석 — 모범 답안과 해설

> 이 문서는 실습 제출 후 공개한다. 제출 전에는 열람하지 않는다.

---

## 실습 1 해설: Kaplan-Meier 생존 곡선과 로그순위 검정

### 제공 코드 실행 결과 해설

Rossi 재범 데이터(432명, 52주 추적)에 Kaplan-Meier 분석을 수행한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 전체 관측치 | 432명 | 출소자 추적 연구 |
| 사건 발생(재범) | 114명 (26.4%) | 약 1/4이 52주 내 재범 |
| 중도절단 | 318명 (73.6%) | 대부분 관측 기간 내 재범하지 않음 |
| 52주 생존 확률 | 0.736 | 52주까지 약 74%가 재범하지 않고 남아 있음 |

주요 시점별 생존 확률:

| 시점(주) | 전체 | 재정 지원 O | 재정 지원 X |
| -------- | ---- | ----------- | ----------- |
| 12 | 0.956 | 0.958 | 0.954 |
| 26 | 0.875 | 0.898 | 0.852 |
| 39 | 0.812 | 0.838 | 0.787 |
| 52 | 0.736 | 0.778 | 0.694 |

로그순위 검정: 검정 통계량 3.838, p-value 0.0501. 유의수준 0.05 경계선으로, 재정 지원의 효과가 있을 가능성을 시사하지만 통계적으로 엄격하게 기각하기 어렵다.

핵심 코드 구조:

```python
# KaplanMeierFitter: event_observed=1이면 사건, 0이면 중도절단
kmf = KaplanMeierFitter()
kmf.fit(durations=data['week'], event_observed=data['arrest'])

# 로그순위 검정: 두 그룹의 전체 추적 기간에 걸친 생존 경험 비교
results = logrank_test(fin_yes['week'], fin_no['week'],
                       fin_yes['arrest'], fin_no['arrest'])
```

생존 곡선이 계단식인 이유: 비모수적 방법이므로 사건 발생 시점에서만 생존 확률이 갱신된다. 두 사건 사이에서는 생존 확률이 일정하게 유지된다.

### 프롬프트 1 모범 구현: 나이 그룹별 생존 곡선 비교

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.datasets import load_rossi
from lifelines.statistics import logrank_test

data = load_rossi()

# 나이 그룹 분리
young = data[data['age'] < 25]
old = data[data['age'] >= 25]
print(f"25세 미만: {len(young)}명 (재범 {young['arrest'].sum()}명)")
print(f"25세 이상: {len(old)}명 (재범 {old['arrest'].sum()}명)")

# 각 그룹 KM 추정
kmf_young = KaplanMeierFitter()
kmf_young.fit(young['week'], young['arrest'], label='Age < 25')

kmf_old = KaplanMeierFitter()
kmf_old.fit(old['week'], old['arrest'], label='Age >= 25')

# 주요 시점 생존 확률
time_points = [12, 26, 52]
print(f"\n{'시점(주)':<10} {'25세 미만':<12} {'25세 이상':<12}")
print("-" * 34)
for t in time_points:
    s_young = kmf_young.survival_function_at_times(t).values[0]
    s_old = kmf_old.survival_function_at_times(t).values[0]
    print(f"{t:<10} {s_young:<12.3f} {s_old:<12.3f}")

# 로그순위 검정
results = logrank_test(young['week'], old['week'],
                       young['arrest'], old['arrest'])
print(f"\n로그순위 검정 p-value: {results.p_value:.4f}")
if results.p_value < 0.05:
    print("결론: 두 그룹의 생존 함수가 통계적으로 유의하게 다름")
else:
    print("결론: 통계적으로 유의한 차이 없음")

# 시각화
fig, ax = plt.subplots(figsize=(8, 5))
kmf_young.plot_survival_function(ax=ax, color='black', linestyle='-')
kmf_old.plot_survival_function(ax=ax, color='gray', linestyle='--')
ax.set_xlabel('Time (weeks)')
ax.set_ylabel('Survival probability')
ax.set_title(f'KM Survival Curves by Age Group (Log-rank p={results.p_value:.4f})')
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig("km_age_group.png", dpi=150)
plt.show()
```

기대 결과 해석:
- 25세 이상 그룹이 전 구간에서 생존 확률이 높을 것으로 예상. 나이가 많을수록 재범 위험이 낮다는 Cox 모형의 결과(HR=0.944)와 일치
- 52주 시점에서 25세 이상 그룹의 생존 확률이 25세 미만보다 높음
- 로그순위 검정에서 p-value < 0.05일 가능성이 높으나, 나이를 이분화하면 정보 손실이 발생하므로 Cox 모형처럼 연속변수로 분석하는 것이 더 효율적

---

## 실습 2 해설: Cox 비례위험 모형으로 재범 위험 요인 분석

### 제공 코드 실행 결과 해설

Rossi 재범 데이터에 Cox 비례위험 모형을 적합한 결과:

| 변수 | 계수 | 위험비 | 95% CI | p-value | 해석 |
| ---- | ---- | ------ | ------ | ------- | ---- |
| fin | -0.379 | 0.684 | (0.470, 0.996) | 0.047* | 재정 지원 → 31.6% 위험 감소 |
| age | -0.057 | 0.944 | (0.904, 0.986) | 0.009** | 1세 증가 → 5.6% 위험 감소 |
| race | 0.314 | 1.369 | (0.748, 2.503) | 0.308 | 통계적으로 유의하지 않음 |
| wexp | -0.150 | 0.861 | (0.568, 1.305) | 0.480 | 통계적으로 유의하지 않음 |
| mar | -0.434 | 0.648 | (0.307, 1.370) | 0.256 | 통계적으로 유의하지 않음 |
| paro | -0.085 | 0.919 | (0.626, 1.348) | 0.665 | 통계적으로 유의하지 않음 |
| prio | 0.091 | 1.096 | (1.036, 1.159) | 0.001** | 전과 1회 → 9.6% 위험 증가 |

C-index 0.640: 두 대상 중 누가 먼저 재범하는지를 64% 확률로 맞춤. 보통 수준의 예측력.

핵심 코드 구조:

```python
# CoxPHFitter: duration_col과 event_col만 지정하면 나머지는 모두 공변량
cph = CoxPHFitter()
cph.fit(data, duration_col='week', event_col='arrest')

# 위험비: exp(coef) — 1보다 크면 위험 증가, 작으면 감소
# 95% CI가 1을 포함하면 통계적으로 유의하지 않음
```

### 프롬프트 2 모범 구현: 전과 횟수 그룹별 예측 생존 곡선

```python
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi

data = load_rossi()

# Cox 모형 적합
cph = CoxPHFitter()
cph.fit(data, duration_col='week', event_col='arrest')

# 전과 횟수별 예측 생존 곡선
fig, ax = plt.subplots(figsize=(8, 5))
cph.plot_partial_effects_on_outcome(
    covariates='prio',
    values=[0, 3, 6, 10],
    ax=ax,
    cmap='gray'
)
ax.set_title('Predicted Survival Curves by Prior Convictions')
ax.set_xlabel('Time (weeks)')
ax.set_ylabel('Survival probability')
ax.legend(title='Prior convictions', labels=['0', '3', '6', '10'])
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("cox_prio_survival.png", dpi=150)
plt.show()

# 52주 시점 생존 확률
print(f"\n{'전과 횟수':<12} {'52주 생존 확률'}")
print("-" * 28)
for prio_val in [0, 3, 6, 10]:
    # 평균 공변량에서 prio만 변경
    mean_covariates = data.drop(columns=['week', 'arrest']).mean().to_dict()
    mean_covariates['prio'] = prio_val
    pred_df = pd.DataFrame([mean_covariates])
    surv = cph.predict_survival_function(pred_df, times=[52])
    prob = surv.values[0][0]
    print(f"{prio_val:<12} {prob:.3f}")
```

기대 결과:
- 전과 0회: 52주 생존 확률이 가장 높음 (약 0.80~0.85)
- 전과 10회: 52주 생존 확률이 가장 낮음 (약 0.55~0.65)
- 전과가 많을수록 생존 곡선이 아래로 내려감. 위험비 1.096이 누적되어 효과가 커짐
- HR=1.096이므로, 전과 10회인 사람의 위험 = 전과 0회 대비 1.096^10 ≈ 2.5배

---

## 실습 3 해설: XGBoost AFT와 Random Survival Forest

### 제공 코드 실행 결과 해설

Rossi 데이터(432명, 7개 변수)에 ML 모형을 적용한 결과:

| 모형 | C-index 경향 | 이유 |
| ---- | ------------ | ---- |
| XGBoost AFT | 0.62~0.63 | 소규모 데이터에서 부스팅의 이점 제한적 |
| Random Survival Forest | 0.63~0.64 | 앙상블 효과로 약간 안정적 |

핵심 코드 구조:

```python
# XGBoost AFT: 중도절단을 lower/upper bound로 표현
y_lower = time.copy().astype(float)
y_upper = np.where(event == 1, time, np.inf).astype(float)
# event=1: lower=upper=time (정확한 관측)
# event=0: lower=time, upper=inf (중도절단, 실제 사건은 time 이후)
```

### 프롬프트 3 모범 구현: Cox vs ML 모형 C-index 비교

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

data = load_rossi()
feature_cols = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']

# 데이터 분할
train_idx, test_idx = train_test_split(
    range(len(data)), test_size=0.2, random_state=42,
    stratify=data['arrest']
)
train_data = data.iloc[train_idx]
test_data = data.iloc[test_idx]

# 1. Cox PH
cph = CoxPHFitter()
cph.fit(train_data, duration_col='week', event_col='arrest')
cox_risk = cph.predict_partial_hazard(test_data[feature_cols]).values.flatten()
cox_c = concordance_index_censored(
    test_data['arrest'].astype(bool), test_data['week'], cox_risk
)[0]

# 2. XGBoost AFT
X_train = train_data[feature_cols].values
X_test = test_data[feature_cols].values
time_train = train_data['week'].values.astype(float)
event_train = train_data['arrest'].values

dtrain = xgb.DMatrix(X_train)
dtrain.set_float_info('label_lower_bound', time_train)
dtrain.set_float_info('label_upper_bound',
    np.where(event_train == 1, time_train, np.inf))

params = {
    'objective': 'survival:aft', 'eval_metric': 'aft-nloglik',
    'aft_loss_distribution': 'normal', 'aft_loss_distribution_scale': 1.2,
    'tree_method': 'hist', 'learning_rate': 0.05, 'max_depth': 3,
    'min_child_weight': 5, 'seed': 42
}
xgb_model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
xgb_pred = xgb_model.predict(xgb.DMatrix(X_test))
xgb_c = concordance_index_censored(
    test_data['arrest'].astype(bool), test_data['week'], -xgb_pred
)[0]

# 3. Random Survival Forest
y_train = np.array([(bool(e), t) for e, t in
    zip(train_data['arrest'], train_data['week'])],
    dtype=[('event', bool), ('time', float)])

rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
    min_samples_leaf=5, random_state=42, n_jobs=-1)
rsf.fit(X_train, y_train)
rsf_risk = rsf.predict(X_test)
rsf_c = concordance_index_censored(
    test_data['arrest'].astype(bool), test_data['week'], rsf_risk
)[0]

# 결과 출력
print(f"{'모형':<30} {'C-index':<10}")
print("-" * 40)
print(f"{'Cox PH':<30} {cox_c:.4f}")
print(f"{'XGBoost AFT':<30} {xgb_c:.4f}")
print(f"{'Random Survival Forest':<30} {rsf_c:.4f}")
```

기대 결과:
- 세 모형의 C-index가 0.62~0.65 범위에서 비슷할 것
- 데이터가 432명, 7개 변수로 소규모이고 관계가 대체로 선형이므로 ML의 이점이 제한적
- 핵심: 데이터가 작고 관계가 단순할 때는 Cox PH가 해석 용이성 측면에서 가장 합리적 선택

---

## 실습 4 해설: DeepSurv 딥러닝 생존 분석

### 제공 코드 실행 결과 해설

비선형 합성 데이터(N=5,000, 20개 특성)에 DeepSurv를 적용한 결과:

| 지표 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| Cox PH 테스트 C-index | 0.61~0.62 | 비선형 관계를 포착하지 못함 |
| DeepSurv 검증 C-index | 0.66~0.67 | 신경망이 비선형 패턴 학습 |
| DeepSurv 테스트 C-index | 0.69~0.70 | 과적합 없이 일반화 |
| 성능 향상 | +7~8%p | 비선형 효과가 클수록 격차 증가 |

핵심 코드 구조:

```python
# Cox 부분 우도 손실: 사건 발생 시점에서의 상대적 위험 비교
def cox_partial_likelihood_loss(risk_scores, time, event):
    sorted_idx = torch.argsort(time, descending=True)
    risk_sorted = risk_scores[sorted_idx]
    event_sorted = event[sorted_idx]

    hazard_ratio = torch.exp(risk_sorted)
    log_risk = torch.log(torch.cumsum(hazard_ratio, dim=0))

    uncensored_likelihood = risk_sorted - log_risk
    censored_likelihood = uncensored_likelihood * event_sorted

    return -censored_likelihood.sum() / event_sorted.sum()
```

### 프롬프트 4 모범 구현: 은닉층 구조 변경 실험

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from sksurv.metrics import concordance_index_censored

# generate_synthetic_survival_data 함수는 10-5-deepsurv.py에서 가져옴
# (여기서는 간략화를 위해 임포트 형태로 표시)
import sys
sys.path.append('practice/chapter10/code')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# 데이터 생성 (10-5-deepsurv.py의 함수 재사용)
def generate_synthetic_survival_data(n_samples=5000, n_features=20, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    log_hazard = (
        0.5 * np.sin(2 * X[:, 0]) + 0.3 * X[:, 1]**2 +
        0.4 * np.tanh(X[:, 2]) + 0.3 * np.abs(X[:, 3]) +
        0.2 * X[:, 4] * np.sign(X[:, 5]) +
        0.25 * (X[:, 0] * X[:, 1]) +
        0.2 * (X[:, 2] * X[:, 3] * X[:, 4]) +
        0.15 * np.maximum(0, X[:, 6] - 0.5) +
        0.1 * np.exp(-X[:, 7]**2) + 0.1 * X[:, 8]
    )
    baseline_hazard = 0.02
    scale = np.exp(-log_hazard) / baseline_hazard
    shape = 1.5
    survival_time = scale * np.random.weibull(shape, n_samples)
    censoring_time = np.random.exponential(np.median(survival_time) * 1.2, n_samples)
    observed_time = np.minimum(survival_time, censoring_time).astype(np.float32)
    event = (survival_time <= censoring_time).astype(np.float32)
    max_time = np.percentile(observed_time, 99)
    observed_time = np.clip(observed_time, 0.01, max_time)
    feature_cols = [f'x{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_cols)
    df['time'] = observed_time
    df['event'] = event.astype(int)
    return df, feature_cols

data, feature_cols = generate_synthetic_survival_data()
train_df, temp_df = train_test_split(data, test_size=0.3, random_state=SEED,
                                      stratify=data['event'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED,
                                    stratify=temp_df['event'])

scaler = StandardScaler()
X_train = scaler.fit_transform(train_df[feature_cols].values).astype(np.float32)
X_val = scaler.transform(val_df[feature_cols].values).astype(np.float32)
X_test = scaler.transform(test_df[feature_cols].values).astype(np.float32)
time_train = train_df['time'].values.astype(np.float32)
time_test = test_df['time'].values.astype(np.float32)
event_train = train_df['event'].values.astype(np.float32)
event_test = test_df['event'].values.astype(np.float32)

def cox_loss(risk, time, event):
    order = torch.argsort(time, descending=True)
    r = risk[order]
    e = event[order]
    log_cum = torch.log(torch.cumsum(torch.exp(r), dim=0) + 1e-12)
    return -(((r - log_cum) * e).sum() / e.sum().clamp(min=1))

hidden_configs = [[32, 16], [64, 32, 16], [128, 64, 32]]
results = []

for hidden_layers in hidden_configs:
    torch.manual_seed(SEED)

    # 모델 구성
    layers = []
    prev = 20
    for h in hidden_layers:
        layers.extend([nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.2)])
        prev = h
    layers.append(nn.Linear(prev, 1))
    model = nn.Sequential(*layers)

    # 파라미터 수 계산
    n_params = sum(p.numel() for p in model.parameters())

    # 학습
    dataset = TensorDataset(torch.tensor(X_train), torch.tensor(time_train),
                            torch.tensor(event_train))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for epoch in range(100):
        model.train()
        for bx, bt, be in loader:
            optimizer.zero_grad()
            loss = cox_loss(model(bx).squeeze(-1), bt, be)
            loss.backward()
            optimizer.step()

    # 테스트
    model.eval()
    with torch.no_grad():
        test_risk = model(torch.tensor(X_test)).squeeze(-1).numpy()
    c_idx = concordance_index_censored(event_test.astype(bool), time_test, test_risk)[0]

    results.append({
        'hidden': str(hidden_layers),
        'params': n_params,
        'c_index': c_idx
    })
    print(f"은닉층 {str(hidden_layers):<20} 파라미터: {n_params:<8} C-index: {c_idx:.4f}")

print(f"\n{'은닉층 구조':<22} {'파라미터 수':<14} {'테스트 C-index'}")
print("-" * 50)
for r in results:
    print(f"{r['hidden']:<22} {r['params']:<14} {r['c_index']:.4f}")
```

기대 결과:
- [32, 16]: 가장 작은 구조. 비선형 관계를 충분히 표현하지 못할 수 있음
- [64, 32, 16]: 기본 구조. 적절한 용량으로 대부분 좋은 성능
- [128, 64, 32]: 가장 큰 구조. 데이터가 5,000명이면 과적합 가능성 있음
- 핵심: 모델이 커진다고 성능이 항상 좋아지지 않음. 데이터 크기에 맞는 적절한 용량이 중요

### 프롬프트 5 모범 구현: 학습 곡선 시각화

```python
import matplotlib.pyplot as plt

# 10-5-deepsurv.py 실행 후 history 딕셔너리를 활용
# history = {'train_loss': [...], 'val_c_index': [...]}
# cox_val_c_index = 0.60 (예시)

# 실행 결과에서 가져온 값으로 시각화
fig, ax1 = plt.subplots(figsize=(10, 6))

epochs = range(1, len(history['train_loss']) + 1)

# 훈련 손실 (왼쪽 y축)
ax1.plot(epochs, history['train_loss'], 'k-', linewidth=2, label='Train Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cox Partial Likelihood Loss')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 검증 C-index (오른쪽 y축)
ax2 = ax1.twinx()
ax2.plot(epochs, history['val_c_index'], 'k--', linewidth=2, label='Validation C-index')
ax2.axhline(y=cox_val_c_index, color='gray', linestyle=':', linewidth=1.5,
            label=f'Cox PH baseline ({cox_val_c_index:.3f})')
ax2.set_ylabel('C-index')
ax2.legend(loc='lower right')

plt.title('DeepSurv Training Curve')
plt.tight_layout()
plt.savefig("deepsurv_training_curve.png", dpi=150)
plt.show()
```

기대 결과:
- 훈련 손실은 에폭이 진행될수록 감소
- 검증 C-index는 초기에 빠르게 상승한 후 안정화
- Cox PH 기준선을 초과하는 시점에서 DeepSurv의 비선형 학습 효과가 나타남
- 검증 C-index가 떨어지기 시작하면 과적합 신호 → 조기 종료가 중요

---

## 실습 5 해설: 4가지 모형 종합 비교

### 제공 코드 실행 결과 해설

고차원 합성 데이터(N=8,000, 100개 특성, 비선형 구조)에서 4가지 모형을 비교한 결과:

| 모형 | 테스트 C-index 경향 | 학습 시간 경향 | 이유 |
| ---- | ------------------- | -------------- | ---- |
| Cox PH | 0.71~0.72 | 빠름 (2~3초) | 선형 가정으로 비선형 패턴 일부 놓침 |
| XGBoost AFT | 0.72~0.73 | 빠름 (1~2초) | 트리 기반 비선형 포착 |
| Random Survival Forest | 0.71~0.72 | 느림 (수분) | 앙상블이지만 max_depth 제한 |
| DeepSurv | 0.73~0.74 | 보통 (5~10초) | 다층 비선형 패턴 학습 |

### 프롬프트 6 모범 구현: 모형별 C-index 막대 그래프

```python
import numpy as np
import matplotlib.pyplot as plt

# 실행 결과에서 가져온 값 (실제 실행 결과로 대체)
models = ['Cox PH', 'XGBoost\nAFT', 'Random\nSurvival Forest', 'DeepSurv']
c_indices = [0.716, 0.721, 0.719, 0.731]  # 실제 값으로 대체

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, c_indices, color=['gray', 'dimgray', 'darkgray', 'black'],
              edgecolor='black')

# 기준선
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label='Random (0.5)')

# 값 표시
for bar, val in zip(bars, c_indices):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Test C-index')
ax.set_title('Survival Model Comparison')
ax.set_ylim(0.45, 0.80)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("model_comparison_bar.png", dpi=150)
plt.show()
```

기대 결과:
- DeepSurv가 가장 높은 C-index를 보이지만, 다른 모형과의 차이는 크지 않음
- 해석이 중요한 의료 연구에서는 Cox PH가 합리적 선택. 위험비로 변수 효과를 직접 설명할 수 있기 때문
- 예측만 중요하다면 DeepSurv나 XGBoost AFT가 우수

### 프롬프트 7 모범 구현: Rossi 데이터에서 4가지 모형 비교

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
from lifelines import CoxPHFitter
from lifelines.datasets import load_rossi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored
from torch.utils.data import DataLoader, TensorDataset

data = load_rossi()
feature_cols = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']

train_idx, test_idx = train_test_split(
    range(len(data)), test_size=0.2, random_state=42,
    stratify=data['arrest']
)
train_data = data.iloc[train_idx].reset_index(drop=True)
test_data = data.iloc[test_idx].reset_index(drop=True)

X_train = train_data[feature_cols].values.astype(np.float32)
X_test = test_data[feature_cols].values.astype(np.float32)
time_train = train_data['week'].values.astype(np.float32)
time_test = test_data['week'].values.astype(np.float32)
event_train = train_data['arrest'].values.astype(np.float32)
event_test = test_data['arrest'].values.astype(np.float32)

results = {}

# 1. Cox PH
cph = CoxPHFitter()
cph.fit(train_data, duration_col='week', event_col='arrest')
cox_risk = cph.predict_partial_hazard(test_data[feature_cols]).values.flatten()
results['Cox PH'] = concordance_index_censored(
    event_test.astype(bool), time_test, cox_risk)[0]

# 2. XGBoost AFT
dtrain = xgb.DMatrix(X_train)
dtrain.set_float_info('label_lower_bound', time_train.astype(float))
dtrain.set_float_info('label_upper_bound',
    np.where(event_train == 1, time_train, np.inf).astype(float))
params = {'objective': 'survival:aft', 'eval_metric': 'aft-nloglik',
          'aft_loss_distribution': 'normal', 'aft_loss_distribution_scale': 1.2,
          'tree_method': 'hist', 'learning_rate': 0.05, 'max_depth': 3,
          'min_child_weight': 5, 'seed': 42}
xgb_model = xgb.train(params, dtrain, num_boost_round=200, verbose_eval=False)
xgb_pred = xgb_model.predict(xgb.DMatrix(X_test))
results['XGBoost AFT'] = concordance_index_censored(
    event_test.astype(bool), time_test, -xgb_pred)[0]

# 3. Random Survival Forest
y_train_struct = np.array([(bool(e), float(t)) for e, t in
    zip(event_train, time_train)], dtype=[('event', bool), ('time', float)])
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10,
    min_samples_leaf=5, random_state=42, n_jobs=-1)
rsf.fit(X_train, y_train_struct)
rsf_risk = rsf.predict(X_test)
results['Random Survival Forest'] = concordance_index_censored(
    event_test.astype(bool), time_test, rsf_risk)[0]

# 4. DeepSurv
torch.manual_seed(42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = nn.Sequential(
    nn.Linear(7, 32), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(16, 1)
)
dataset = TensorDataset(torch.tensor(X_train_s, dtype=torch.float32),
                        torch.tensor(time_train), torch.tensor(event_train))
loader = DataLoader(dataset, batch_size=64, shuffle=True)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

for epoch in range(100):
    model.train()
    for bx, bt, be in loader:
        optimizer.zero_grad()
        risk = model(bx).squeeze(-1)
        order = torch.argsort(bt, descending=True)
        r, e = risk[order], be[order]
        log_cum = torch.log(torch.cumsum(torch.exp(r), dim=0) + 1e-12)
        loss = -(((r - log_cum) * e).sum() / e.sum().clamp(min=1))
        loss.backward()
        optimizer.step()

model.eval()
with torch.no_grad():
    ds_risk = model(torch.tensor(X_test_s, dtype=torch.float32)).squeeze(-1).numpy()
results['DeepSurv'] = concordance_index_censored(
    event_test.astype(bool), time_test, ds_risk)[0]

# 결과 출력
print(f"{'모형':<30} {'C-index (Rossi)'}")
print("-" * 45)
for name, c in results.items():
    print(f"{name:<30} {c:.4f}")
```

기대 결과:
- Rossi 데이터(432명, 7변수)에서는 4가지 모형의 C-index가 0.62~0.66 범위에서 비슷
- 합성 데이터와 달리 DeepSurv가 압도적이지 않음. 오히려 과적합 위험
- 데이터가 작고 관계가 단순할 때: Cox PH가 해석 가능성 + 안정적 성능으로 가장 합리적

### 프롬프트 8 모범 구현: 학습 데이터 크기에 따른 성능 변화

```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter
from torch.utils.data import DataLoader, TensorDataset

# 합성 데이터 생성 (큰 풀에서 서브샘플링)
def generate_data(n=6000, n_features=20, seed=42):
    np.random.seed(seed)
    X = np.random.randn(n, n_features).astype(np.float32)
    log_hazard = (
        0.5 * np.sin(2 * X[:, 0]) + 0.3 * X[:, 1]**2 +
        0.4 * np.tanh(X[:, 2]) + 0.3 * np.abs(X[:, 3]) +
        0.25 * (X[:, 0] * X[:, 1]) + 0.2 * (X[:, 2] * X[:, 3] * X[:, 4]) +
        0.1 * X[:, 8]
    )
    scale = np.exp(-log_hazard) / 0.02
    survival_time = scale * np.random.weibull(1.5, n)
    censoring_time = np.random.exponential(np.median(survival_time) * 1.2, n)
    time = np.minimum(survival_time, censoring_time).astype(np.float32)
    event = (survival_time <= censoring_time).astype(np.float32)
    max_t = np.percentile(time, 99)
    time = np.clip(time, 0.01, max_t)
    cols = [f'x{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=cols)
    df['time'] = time
    df['event'] = event.astype(int)
    return df, cols

data, feat_cols = generate_data(n=6000)

# 고정 테스트셋 1000명
test_df = data.iloc[-1000:].reset_index(drop=True)
pool_df = data.iloc[:-1000].reset_index(drop=True)

X_test = test_df[feat_cols].values.astype(np.float32)
time_test = test_df['time'].values.astype(np.float32)
event_test = test_df['event'].values.astype(np.float32)

train_sizes = [500, 1000, 2000, 4000]
results = []

for n_train in train_sizes:
    torch.manual_seed(42)
    train_df = pool_df.sample(n=n_train, random_state=42).reset_index(drop=True)
    X_train = train_df[feat_cols].values.astype(np.float32)
    time_train = train_df['time'].values.astype(np.float32)
    event_train = train_df['event'].values.astype(np.float32)

    # Cox PH
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_df[feat_cols + ['time', 'event']], duration_col='time', event_col='event')
    cox_risk = cph.predict_partial_hazard(test_df[feat_cols]).values.flatten()
    cox_c = concordance_index_censored(event_test.astype(bool), time_test, cox_risk)[0]

    # DeepSurv
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    model = nn.Sequential(
        nn.Linear(20, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 1)
    )
    dataset = TensorDataset(torch.tensor(X_tr_s, dtype=torch.float32),
                            torch.tensor(time_train), torch.tensor(event_train))
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    for _ in range(100):
        model.train()
        for bx, bt, be in loader:
            optimizer.zero_grad()
            r = model(bx).squeeze(-1)
            order = torch.argsort(bt, descending=True)
            rs, es = r[order], be[order]
            loss = -(((rs - torch.log(torch.cumsum(torch.exp(rs), 0) + 1e-12)) * es).sum()
                     / es.sum().clamp(min=1))
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        ds_risk = model(torch.tensor(X_te_s, dtype=torch.float32)).squeeze(-1).numpy()
    ds_c = concordance_index_censored(event_test.astype(bool), time_test, ds_risk)[0]

    results.append({'n_train': n_train, 'cox_c': cox_c, 'ds_c': ds_c})
    print(f"N={n_train:>5}: Cox={cox_c:.4f}, DeepSurv={ds_c:.4f}")

print(f"\n{'학습 크기':<12} {'Cox PH':<12} {'DeepSurv':<12} {'차이'}")
print("-" * 48)
for r in results:
    diff = r['ds_c'] - r['cox_c']
    print(f"{r['n_train']:<12} {r['cox_c']:<12.4f} {r['ds_c']:<12.4f} {diff:+.4f}")
```

기대 결과:
- N=500: Cox PH가 DeepSurv보다 좋거나 비슷. 데이터가 적으면 신경망이 과적합
- N=1000: 두 모형이 비슷한 수준
- N=2000~4000: DeepSurv가 Cox PH를 추월하기 시작. 비선형 패턴 학습의 이점이 나타남
- 핵심: DeepSurv는 데이터가 충분할 때(수천 건+) 이점이 있고, 적은 데이터에서는 Cox PH가 더 안정적

---

## 10장 전체 핵심 정리

```text
1. 생존 분석의 핵심은 중도절단 처리다. 불완전한 관측을 버리지 않고 활용하여
   편향 없는 생존 함수 추정이 가능하다.
2. Kaplan-Meier는 비모수적 생존 곡선 추정의 기본이다. 로그순위 검정으로
   그룹 간 차이를 비교하되, 다변량 분석에는 Cox 모형이 필요하다.
3. Cox 비례위험 모형의 위험비(HR)는 변수의 효과를 직접 해석할 수 있는 강력한 도구다.
   비례위험 가정 위반 시 AFT 또는 층화 모형을 검토한다.
4. ML 모형(XGBoost AFT, RSF)은 비선형 관계를 포착하지만,
   데이터가 적으면(수백 건) Cox PH와 큰 차이가 없다.
5. DeepSurv는 충분한 데이터(수천 건+)와 비선형 관계가 존재할 때 전통 모형을 초과하지만,
   해석 가능성과 운영 비용의 트레이드오프가 있다.
6. 모형 선택은 C-index 성능만이 아니라 해석 가능성, 학습/운영 비용,
   비례위험 가정 충족 여부를 종합적으로 고려해야 한다.
```
