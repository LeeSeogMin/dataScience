# 5장 B: 앙상블 학습과 모델 해석 — 모범 답안과 해설

> 이 문서는 실습 제출 후 공개한다. 제출 전에는 열람하지 않는다.

---

## 실습 1 해설: 랜덤 포레스트 OOB 평가와 특성 중요도

### 제공 코드 실행 결과 해설

California Housing 데이터(16,512 훈련 / 4,128 테스트, 8특성)에 랜덤 포레스트를 적용한 결과:

| n_estimators | OOB Score | Test RMSE | Test R2 | 학습 시간(초) |
| ------------ | --------- | --------- | ------- | ------------- |
| 50 | 0.8007 | 0.5072 | 0.8037 | ~1.1 |
| 100 | 0.8074 | 0.5053 | 0.8051 | ~1.4 |
| 200 | 0.8100 | 0.5040 | 0.8062 | ~2.9 |
| 300 | 0.8108 | 0.5034 | 0.8066 | ~3.8 |
| 500 | 0.8121 | 0.5022 | 0.8075 | ~6.1 |

핵심 인사이트:
- OOB Score와 Test R2가 유사한 수준을 보여, OOB가 별도 검증 세트 없이도 일반화 성능을 신뢰성 있게 추정함을 확인
- n_estimators가 200-300 이상에서 개선 폭이 매우 작아짐 (수익 체감)
- 특성 중요도: MedInc(0.52) >> AveOccup, HouseAge, Latitude, Longitude 순

### 프롬프트 1 모범 구현: max_features 변경 실험

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"{'max_features':<14} {'OOB Score':<12} {'Test RMSE':<12} {'시간(초)':<10}")
print("-" * 48)

for mf in [0.3, 0.5, 'sqrt', 0.7, 1.0]:
    start = time.time()
    model = RandomForestRegressor(
        n_estimators=300, max_features=mf, oob_score=True,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    label = str(mf)
    print(f"{label:<14} {model.oob_score_:<12.4f} {rmse:<12.4f} {elapsed:<10.2f}")
```

기대 결과 해석:
- max_features=0.3: 트리 간 다양성 최고, 개별 트리 약함. OOB Score가 가장 낮을 수 있음
- max_features='sqrt' (~0.35 for 8 features): 분류의 경험적 최적값. 회귀에서는 p/3이 권장이지만 유사
- max_features=1.0: 모든 특성 사용 = 일반 배깅. 다양성 최저, 개별 트리 강함. 앙상블 효과 감소
- 최적 균형: 일반적으로 0.5~sqrt 범위에서 OOB Score가 가장 높음
- max_features가 작을수록 학습 시간이 짧아짐 (탐색할 특성 후보가 적으므로)

---

## 실습 2 해설: 그래디언트 부스팅 3종 비교

### 제공 코드 실행 결과 해설

동일 하이퍼파라미터(n_estimators=200, max_depth=6, learning_rate=0.1)로 비교한 결과:

| 모델 | RMSE | R2 | 학습 시간(초) |
| ---- | ---- | -- | ------------- |
| XGBoost | 0.4639 | 0.8358 | ~0.4 |
| LightGBM | 0.4528 | 0.8436 | ~0.2 |
| CatBoost | 0.4799 | 0.8243 | ~0.9 |

핵심 인사이트:
- 세 라이브러리의 RMSE 차이가 0.03 수준으로 매우 작음 → 동일 원리의 다른 구현
- LightGBM이 RMSE도 가장 낮고 학습도 가장 빠름 (히스토그램 + 리프 중심 성장의 효과)
- CatBoost는 범주형 특성이 없는 이 데이터에서는 불리. 범주형이 많은 데이터에서 강점 발휘
- 실무에서는 성능 차이보다 운영 환경(속도, 메모리, 범주형 처리)에 따라 선택

### 프롬프트 2 모범 구현: learning_rate 변경 실험

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import time

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"{'learning_rate':<16} {'RMSE':<10} {'시간(초)':<10}")
print("-" * 36)

for lr in [0.01, 0.05, 0.1, 0.2, 0.3]:
    start = time.time()
    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=lr,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"{lr:<16} {rmse:<10.4f} {elapsed:<10.2f}")
```

기대 결과 해석:
- learning_rate=0.01: RMSE가 높음. 200개 트리로는 수렴하지 못함. 각 트리의 기여가 너무 작아 충분히 학습하지 못한 상태(언더피팅)
- learning_rate=0.05~0.1: 적절한 균형. RMSE가 가장 낮은 구간
- learning_rate=0.2~0.3: 각 트리가 너무 많이 기여하여 초기 트리의 오류가 크게 반영됨. 과적합 경향
- 실무 전략: learning_rate를 0.01~0.05로 낮추고 n_estimators를 크게 설정(1000+)한 뒤, 조기 종료로 최적 트리 수를 자동 결정

---

## 실습 3 해설: Optuna 하이퍼파라미터 최적화

### 제공 코드 실행 결과 해설

| 모델 | RMSE | R2 | 개선율(RMSE) |
| ---- | ---- | -- | ------------ |
| XGBoost 기본 | 0.4718 | 0.8301 | - |
| XGBoost 최적화 | ~0.4365 | ~0.8546 | ~7.5% |
| LightGBM 기본 | 0.4635 | 0.8360 | - |
| LightGBM 최적화 | ~0.4378 | ~0.8537 | ~5.6% |

핵심 인사이트:
- 기본 설정 대비 5~7.5% RMSE 개선. 하이퍼파라미터 튜닝의 실질적 가치를 확인
- LightGBM의 기본 성능이 더 좋지만, 튜닝 후에는 XGBoost와 거의 동일
- 최적화 과정 시각화에서 초기 변동 → 후반 수렴 패턴: 베이지안 최적화가 유망 영역을 집중 탐색하는 증거

### 프롬프트 3 모범 구현: 조기 종료 활용

```python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb

data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 학습 세트를 다시 학습/검증으로 분할
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=42
)

model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
    eval_metric="rmse",
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False
)

y_pred = model.predict(X_test)
test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

print(f"설정 n_estimators: 1000")
print(f"best_iteration: {model.best_iteration}")
print(f"테스트 RMSE: {test_rmse:.4f}")
```

기대 결과 해석:
- best_iteration이 200-400 정도로, 설정한 1000보다 훨씬 작음
- 이는 50라운드 동안 검증 RMSE가 개선되지 않아 자동 중단된 것
- 조기 종료의 장점: n_estimators를 크게 잡아도 과적합 없이 최적 시점에 멈춤
- 즉, n_estimators를 직접 튜닝하는 대신 조기 종료가 자동으로 최적 트리 수를 결정
- best_iteration이 n_estimators에 가까우면 n_estimators를 더 늘려야 할 수 있음

---

## 실습 4 해설: 모델 해석 (SHAP, PDP, Permutation Importance)

### 제공 코드 실행 결과 해설

XGBoost(n_estimators=400, max_depth=4, learning_rate=0.05)로 California Housing 데이터를 학습한 결과:

| 항목 | 값 경향 |
| ---- | ------- |
| 테스트 RMSE | ~0.46 |
| 테스트 R2 | ~0.84 |
| SHAP 전역 중요도 Top-1 | Latitude |

특성 중요도 비교:

| 특성 | SHAP | MDI (Gain) | Permutation |
| ---- | ---- | ---------- | ----------- |
| Latitude | 0.454 | 0.089 | 0.985 |
| MedInc | 0.408 | 0.498 | 0.463 |
| Longitude | 0.402 | 0.092 | 0.861 |
| AveOccup | 0.205 | 0.126 | 0.145 |
| AveRooms | 0.095 | 0.081 | 0.044 |

핵심 인사이트:
- MDI(Gain)에서는 MedInc가 1위(0.498)이지만, SHAP과 Permutation에서는 Latitude가 1위
- MDI는 분할 빈도에 편향: MedInc으로 분할할 때 불순도 감소가 크므로 MDI가 높지만, 실제 예측 기여도는 위치(Latitude, Longitude)가 더 큼
- 실무에서는 세 방법을 함께 확인하여 일관된 패턴을 찾는 것이 안전

### 프롬프트 4 모범 구현: SHAP 의존도 분석

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

X_sample = X_test.sample(n=1000, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

# MedInc에 대한 dependence plot (상호작용 특성 자동 선택)
plt.figure(figsize=(8, 5))
shap.dependence_plot("MedInc", shap_values, X_sample, show=False)
plt.title("SHAP Dependence: MedInc")
plt.tight_layout()
plt.savefig("shap_dependence_medinc.png", dpi=150)
plt.show()

# 상호작용 특성 확인
interaction_idx = shap.approximate_interactions("MedInc", shap_values, X_sample)
print(f"MedInc와 가장 강한 상호작용 특성: {X_sample.columns[interaction_idx[0]]}")
```

기대 결과 해석:
- MedInc가 증가할 때 SHAP 값도 증가하는 양의 상관 패턴 (소득 높으면 집값 예측 올라감)
- MedInc > 6 구간에서 SHAP 값의 증가가 더 가파름 (고소득 지역의 가격 프리미엄)
- 색상으로 표시된 상호작용 특성(주로 AveOccup 또는 Latitude): 같은 소득 수준이라도 위치에 따라 집값 효과가 다름
- 이는 "중위소득이 높더라도 북부 지역은 남부 지역보다 집값이 낮다"는 상호작용을 반영

### 프롬프트 5 모범 구현: 개별 예측 해석 (Waterfall Plot)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import xgboost as xgb
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

data = fetch_california_housing(as_frame=True)
X, y = data.data, data.target
feature_names = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
model.fit(X_train, y_train)

explainer = shap.TreeExplainer(model)
X_sample = X_test.iloc[:5]
shap_values = explainer.shap_values(X_sample)
base_value = explainer.expected_value

print(f"{'샘플':<8} {'예측값':<10} {'Top-1 특성':<14} {'Top-2 특성':<14} {'Top-3 특성':<14}")
print("-" * 60)

for i in range(5):
    pred = model.predict(X_sample.iloc[[i]])[0]
    abs_shap = np.abs(shap_values[i])
    top3_idx = np.argsort(abs_shap)[-3:][::-1]
    top3 = [f"{feature_names[j]}({shap_values[i][j]:+.2f})" for j in top3_idx]
    print(f"{i:<8} {pred:<10.3f} {top3[0]:<14} {top3[1]:<14} {top3[2]:<14}")

    # Waterfall plot 생성
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[i],
            base_values=base_value,
            data=X_sample.iloc[i],
            feature_names=feature_names
        ),
        show=False
    )
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_sample{i}.png", dpi=150)
    plt.close()

print(f"\nbase value (전체 평균 예측): {base_value:.3f}")
```

기대 결과 해석:
- 각 샘플의 예측값이 base value(전체 평균)에서 시작해 각 특성의 기여도가 순차적으로 누적
- MedInc가 높은 샘플은 양의 SHAP 기여, 낮은 샘플은 음의 기여
- Latitude가 낮은(남부) 샘플은 집값을 올리는 방향으로 기여
- 같은 특성이라도 샘플마다 기여 방향과 크기가 다를 수 있음 → 개별 예측 해석의 가치

---

## 실습 5 해설: DiCE 반사실적 설명

### 제공 코드 실행 결과 해설

Breast Cancer 데이터에서 XGBoost 분류기 성능:

| 항목 | 값 경향 |
| ---- | ------- |
| Accuracy | ~0.956 |
| ROC AUC | ~0.995 |

counterfactual 생성: 현재 malignant(0)로 예측된 샘플을 benign(1)으로 바꾸기 위한 변화량:

| cf_id | 주요 변경 특성 | 원본값 | counterfactual 값 | 변화량 |
| ----- | -------------- | ------ | ----------------- | ------ |
| 0 | worst_area | 1926.0 | ~231.9 | -1694.1 |
| 0 | mean_texture | 28.77 | ~10.14 | -18.63 |
| 0 | worst_concave_points | 0.1941 | ~0.0 | -0.1941 |

핵심 인사이트:
- worst_area(최대 면적)가 가장 큰 변화를 요구: 악성 종양의 크기가 판정에 결정적
- 3개의 counterfactual이 서로 다른 특성 조합을 변경: 같은 결과를 달성하는 여러 경로가 존재
- SHAP과의 보완 관계: SHAP은 "worst_area가 예측에 기여한 정도"를 보여주고, counterfactual은 "worst_area를 얼마나 바꿔야 결과가 바뀌는지"를 보여줌

### 프롬프트 6 모범 구현: 특정 특성 고정 counterfactual

```python
import re
import warnings
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import dice_ml

warnings.filterwarnings("ignore")

def sanitize_column(name):
    name = re.sub(r"[^A-Za-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df = df.rename(columns={c: sanitize_column(c) for c in df.columns})
feature_cols = [c for c in df.columns if c != "target"]

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])

model = xgb.XGBClassifier(
    n_estimators=400, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
    eval_metric="logloss", n_jobs=-1
)
model.fit(train_df[feature_cols], train_df["target"])

# 예측이 0(malignant)인 샘플 선택
test_pred = model.predict(test_df[feature_cols])
idx_candidates = np.where(test_pred == 0)[0]
query_row = test_df.iloc[[int(idx_candidates[0])]].copy()
query_x = query_row[feature_cols]

# worst_area, worst_perimeter를 제외한 변경 가능 특성 목록
features_to_vary = [f for f in feature_cols if f not in ["worst_area", "worst_perimeter"]]

dice_data = dice_ml.Data(dataframe=train_df, continuous_features=feature_cols, outcome_name="target")
dice_model = dice_ml.Model(model=model, backend="sklearn")
dice = dice_ml.Dice(dice_data, dice_model, method="random")

exp = dice.generate_counterfactuals(
    query_x,
    total_CFs=5,
    desired_class="opposite",
    features_to_vary=features_to_vary,
    random_seed=42
)

cf_df = exp.cf_examples_list[0].final_cfs_df
print("Counterfactual 결과 (worst_area, worst_perimeter 고정):\n")

# 변화량 분석
for i, row in cf_df.iterrows():
    changes = []
    for col in feature_cols:
        orig = query_x.iloc[0][col]
        cf_val = row[col]
        if abs(cf_val - orig) > 1e-6:
            changes.append((col, orig, cf_val, cf_val - orig))

    print(f"--- CF {i} ---")
    if changes:
        for col, orig, cf_val, delta in sorted(changes, key=lambda x: abs(x[3]), reverse=True)[:5]:
            print(f"  {col}: {orig:.4f} → {cf_val:.4f} (변화량: {delta:+.4f})")
    print()
```

기대 결과 해석:
- worst_area와 worst_perimeter가 고정되어 있으므로, 다른 특성(mean_texture, mean_concavity 등)으로 보상해야 함
- 제약이 있는 counterfactual은 변경해야 할 특성이 더 많고 변화량도 더 클 수 있음
- 실무 적용: 대출 심사에서 "나이"는 바꿀 수 없으므로 "소득"이나 "신용점수"만으로 승인 조건을 제시
- 5개의 다양한 counterfactual이 서로 다른 경로를 보여줌: 의사결정자에게 여러 옵션을 제공

---

## 실습 6 해설: LLM 임베딩 + XGBoost 하이브리드

### 제공 코드 실행 결과 해설

이커머스 가상 데이터(2,000샘플, 구매율 ~30%)에서:

| 입력 구성 | Accuracy | ROC AUC | F1 Score |
| --------- | -------- | ------- | -------- |
| 정형 특성만 | 0.6600 | 0.5579 | 0.1905 |
| 정형+임베딩(하이브리드) | 0.6475 | 0.6452 | 0.2985 |
| 변화량 | -0.0125 | +0.0872 | +0.1080 |

핵심 인사이트:
- Accuracy는 소폭 하락했지만 ROC AUC(+8.7%)와 F1(+56.7%)은 의미 있게 개선
- 텍스트에 담긴 감정(긍정 리뷰 → 재구매), 의도(할인 검색 → 구매), 관심(배송 문의 → 구매)이 임베딩을 통해 반영
- 불균형 데이터에서 accuracy보다 ROC AUC/F1이 더 적절한 평가 지표
- PCA 384→50으로 축소해도 분산 97.8% 보존: 임베딩 정보 손실 최소

### 프롬프트 7 모범 구현: PCA 차원 수 변경 실험

```python
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")
np.random.seed(42)

# 데이터 생성 함수 (5-8-llm-xgboost.py에서 가져옴)
# generate_ecommerce_data, build_text, build_tabular_matrices 함수를 재사용
# (실습에서는 5-8-llm-xgboost.py를 import하거나 코드를 복사)

# 아래는 핵심 실험 부분만 표시
# df = generate_ecommerce_data(n_samples=2000)
# ... (데이터 준비, 임베딩 추출 완료 후)

# 이미 emb_train, emb_test, X_tab_train, X_tab_test, y_train, y_test가 준비된 상태에서:
n_components_list = [10, 20, 50, 100, 200]

print(f"{'n_components':<14} {'분산설명':<10} {'Accuracy':<12} {'ROC AUC':<12} {'F1 Score':<12}")
print("-" * 60)

for n_comp in n_components_list:
    pca = PCA(n_components=n_comp, random_state=42)
    emb_train_r = pca.fit_transform(emb_train)
    emb_test_r = pca.transform(emb_test)
    explained = pca.explained_variance_ratio_.sum()

    X_hyb_train = np.hstack([X_tab_train, emb_train_r])
    X_hyb_test = np.hstack([X_tab_test, emb_test_r])

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", n_jobs=-1
    )
    model.fit(X_hyb_train, y_train)

    proba = model.predict_proba(X_hyb_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)

    print(f"{n_comp:<14} {explained:<10.4f} {acc:<12.4f} {auc:<12.4f} {f1:<12.4f}")
```

기대 결과 해석:
- n_components=10: 분산 설명 ~80%. 텍스트 정보 손실이 커서 개선 폭이 작음
- n_components=20~50: 분산 설명 90%+. 적절한 균형점. F1과 ROC AUC 개선이 뚜렷
- n_components=100~200: 분산 설명 99%+. 차원이 많아지면서 과적합 위험 증가
- 정형 특성(13개)보다 임베딩 차원이 훨씬 많으면 XGBoost가 임베딩 위주로 학습할 수 있음 → 차원 축소 필수
- 최적 차원: 일반적으로 50 전후가 실무에서 자주 사용됨 (분산 95%+ 보존하면서 과적합 방지)

### 프롬프트 8 모범 구현: 텍스트 특성별 기여도 분석

```python
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.decomposition import PCA
import xgboost as xgb
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# 데이터 생성 및 분할 완료 후...
# train_df, test_df, X_tab_train, X_tab_test, y_train, y_test 준비된 상태

embedder = SentenceTransformer("all-MiniLM-L6-v2")
n_components = 50

text_configs = [
    ("review_text만", ["review_text"]),
    ("inquiry_text만", ["inquiry_text"]),
    ("search_keywords만", ["search_keywords"]),
    ("세 가지 모두", ["review_text", "inquiry_text", "search_keywords"]),
]

print(f"{'텍스트 구성':<20} {'Accuracy':<12} {'ROC AUC':<12} {'F1 Score':<12}")
print("-" * 56)

for label, text_cols in text_configs:
    # 텍스트 결합
    train_text = train_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
    test_text = test_df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()

    # 임베딩 추출 및 PCA
    emb_train = embedder.encode(train_text, show_progress_bar=False)
    emb_test = embedder.encode(test_text, show_progress_bar=False)

    pca = PCA(n_components=n_components, random_state=42)
    emb_train_r = pca.fit_transform(emb_train)
    emb_test_r = pca.transform(emb_test)

    # 하이브리드 모델
    X_hyb_train = np.hstack([X_tab_train, emb_train_r])
    X_hyb_test = np.hstack([X_tab_test, emb_test_r])

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", n_jobs=-1
    )
    model.fit(X_hyb_train, y_train)

    proba = model.predict_proba(X_hyb_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, pred)

    print(f"{label:<20} {acc:<12.4f} {auc:<12.4f} {f1:<12.4f}")
```

기대 결과 해석:
- review_text가 가장 큰 기여: 긍정/부정 감정이 재구매 의향과 직접 연결
- search_keywords도 의미 있는 기여: "할인", "특가" 같은 구매 의도 키워드가 포착됨
- inquiry_text는 기여가 상대적으로 적을 수 있음: 문의 내용은 구매 의향보다 서비스 관련
- 세 가지를 모두 결합하면 가장 좋은 성능: 서로 다른 유형의 텍스트 신호가 보완적으로 작용
- 실무 시사점: 모든 텍스트를 무조건 결합하기보다, 텍스트별 기여도를 확인하고 비용 대비 가치가 있는 텍스트만 사용하는 것이 효율적

---

## 5장 전체 핵심 정리

```text
1. 랜덤 포레스트: max_features로 트리 간 다양성과 개별 트리 성능의 균형을 조절한다.
   OOB 평가는 별도 검증 세트 없이 일반화 성능을 추정하는 효율적 방법이다.
2. 그래디언트 부스팅: learning_rate를 낮추면 더 정밀하게 학습하지만 더 많은 트리가 필요하다.
   조기 종료로 최적 트리 수를 자동 결정하는 것이 실무 표준이다.
3. Optuna: 베이지안 최적화로 기본 설정 대비 5~7% 성능 개선이 가능하다.
   파라미터 중요도를 확인하여 중요한 파라미터에 집중하는 것이 효율적이다.
4. 모델 해석: SHAP, MDI, Permutation의 중요도 순위가 다를 수 있다.
   세 방법을 함께 확인하여 일관된 패턴을 찾는 것이 안전하다.
5. Counterfactual: "무엇을 바꾸면 결과가 달라지는가"는 SHAP의 "왜"를 보완한다.
   실무에서는 조정 가능한 변수와 불가능한 변수를 구분하여 실현 가능한 제안을 만든다.
6. LLM 하이브리드: 텍스트 임베딩은 정형 특성이 못 포착하는 감정·의도 신호를 제공한다.
   PCA 차원 축소로 과적합을 방지하되, 최적 차원은 데이터에 따라 실험으로 결정한다.
7. AI 도구로 코드를 생성하되, 결과를 반드시 검증하고 해석하는 습관이 중요하다.
```
