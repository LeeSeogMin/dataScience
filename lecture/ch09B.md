# 9장 B: 시계열 분석과 예측 — 모범 답안과 해설

> 이 문서는 실습 제출 후 공개한다. 제출 전에는 열람하지 않는다.

---

## 실습 1 해설: SARIMA로 CO2 농도 예측

### 제공 코드 실행 결과 해설

Mauna Loa CO2 데이터(1958~2001년, 월별 526개)에 SARIMA(0,1,1)(0,1,1)12 모형을 적용한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 원본 p-value | 0.999 | 강한 상승 추세로 인해 비정상 |
| 1차 차분 p-value | 0.0001 미만 | 추세 제거 후 정상 |
| AIC | 195.82 전후 | 파라미터 3개의 단순 모형 |
| RMSE | 2.30 ppm | 360ppm 수준에서 약 2ppm 오차 |
| MAPE | 0.53% | 매우 정확한 예측 |

핵심 코드 구조:

```python
# SARIMA(0,1,1)(0,1,1)_12 모형의 의미:
# - p=0: AR 항 없음 (과거 값에 직접 의존하지 않음)
# - d=1: 1차 차분으로 추세 제거
# - q=1: 직전 시점 오차 반영 (MA(1) 계수 θ₁ = -0.356)
# - P=0, D=1, Q=1, m=12: 계절 차분 + 12개월 전 오차 반영 (Θ₁ = -0.842)
model = SARIMAX(y_train, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False)
```

MA(1) 계수 θ₁ = -0.356은 직전 시점의 예측 오차가 현재 값에 음의 방향으로 작용하여, 과도한 예측 조정을 완화하는 역할을 한다. 계절 MA(1) 계수 Θ₁ = -0.842가 큰 이유는 CO2 농도의 강한 연간 계절성(북반구 식물 광합성 주기: 봄에 최고, 가을에 최저)을 반영하기 때문이다.

### 프롬프트 1 모범 구현: ARIMA 차수 변경 실험

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

data_path = Path(__file__).parent.parent / "data" / "co2.csv"
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
y = df['co2']

train_size = int(len(y) * 0.8)
y_train = y[:train_size]
y_test = y[train_size:]

orders = [(0, 1, 1), (1, 1, 0), (1, 1, 1)]
seasonal_order = (0, 1, 1, 12)

print(f"{'order':<14} {'AIC':<12} {'BIC':<12} {'RMSE':<12} {'MAPE(%)':<10}")
print("-" * 60)

for order in orders:
    model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    forecast = result.get_forecast(steps=len(y_test)).predicted_mean
    rmse = np.sqrt(mean_squared_error(y_test, forecast))
    mape = mean_absolute_percentage_error(y_test, forecast) * 100
    print(f"{str(order):<14} {result.aic:<12.2f} {result.bic:<12.2f} {rmse:<12.4f} {mape:<10.2f}")
```

기대 결과 해석:
- (0,1,1): 가장 단순한 모형. AIC가 낮고 RMSE도 안정적
- (1,1,0): AR(1) 모형. 과거 값에 직접 의존하는 구조
- (1,1,1): AR+MA 결합. 파라미터가 많지만 반드시 성능이 좋아지지는 않음
- 핵심: AIC가 가장 낮은 모형이 예측 성능도 가장 좋은 경향이 있지만, 항상 그런 것은 아니다. AIC는 모형의 적합도와 복잡도 사이의 균형이고, RMSE는 외삽(out-of-sample) 성능이다

### 프롬프트 2 모범 구현: 예측 기간별 성능 변화

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

data_path = Path(__file__).parent.parent / "data" / "co2.csv"
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
y = df['co2']

train_size = int(len(y) * 0.8)
y_train = y[:train_size]
y_test = y[train_size:]

model = SARIMAX(y_train, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
                enforce_stationarity=False, enforce_invertibility=False)
result = model.fit(disp=False)
forecast = result.get_forecast(steps=len(y_test)).predicted_mean

periods = [12, 24, 48, len(y_test)]
print(f"{'예측 기간':<14} {'RMSE':<12} {'MAPE(%)':<10}")
print("-" * 36)

for n in periods:
    actual = y_test.values[:n]
    pred = forecast.values[:n]
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mean_absolute_percentage_error(actual, pred) * 100
    label = f"{n}개월" if n < len(y_test) else f"전체({n})"
    print(f"{label:<14} {rmse:<12.4f} {mape:<10.2f}")
```

기대 결과:
- 12개월: 가장 낮은 RMSE. 단기 예측은 정확
- 48개월, 전체: RMSE가 점차 증가. 추세 외삽의 오차가 시간에 따라 누적
- 핵심: 예측 기간이 길어질수록 불확실성이 증가한다. 이것이 95% 신뢰구간이 넓어지는 이유

---

## 실습 2 해설: Prophet으로 CO2 농도 예측

### 제공 코드 실행 결과 해설

동일한 CO2 데이터에 Prophet을 적용한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 가법 RMSE | 0.87 ppm | 추세+계절성 분리가 효과적 |
| 승법 RMSE | 0.87 ppm | CO2 계절 변동이 추세와 무관하게 일정 |
| 변화점 수 | 25개 전후 | CO2 증가율의 미세한 변화 포착 |

핵심 코드 구조:

```python
# Prophet 형식 변환: ds(날짜)와 y(값) 컬럼 필요
prophet_df = pd.DataFrame({'ds': y.index, 'y': y.values})

# 가법 모형과 승법 모형 비교
model_add = Prophet(seasonality_mode='additive', changepoint_prior_scale=0.05)
model_mult = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.05)
```

Prophet이 SARIMA보다 좋은 성능을 보인 이유: CO2 데이터는 완만한 상승 추세와 규칙적인 연간 계절성이라는 두 가지 명확한 구성 요소로 되어 있다. Prophet은 이 두 요소를 명시적으로 분리하여 각각 정확하게 추정하고 결합한다. SARIMA는 차분을 통해 암묵적으로 처리하므로 약간의 정보 손실이 발생한다.

### 프롬프트 3 모범 구현: changepoint_prior_scale 변경 실험

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

data_path = Path(__file__).parent.parent / "data" / "co2.csv"
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
y = df['co2']

prophet_df = pd.DataFrame({'ds': y.index, 'y': y.values})
train_size = int(len(prophet_df) * 0.8)
df_train = prophet_df[:train_size]
df_test = prophet_df[train_size:]

scales = [0.01, 0.05, 0.1, 0.5]

print(f"{'scale':<10} {'RMSE':<12} {'MAPE(%)':<10} {'변화점 수':<10}")
print("-" * 42)

for scale in scales:
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=scale
    )
    model.fit(df_train)

    future = model.make_future_dataframe(periods=len(df_test), freq='MS')
    forecast = model.predict(future)
    forecast_test = forecast[forecast['ds'] >= df_test['ds'].iloc[0]]

    rmse = np.sqrt(mean_squared_error(df_test['y'].values, forecast_test['yhat'].values))
    mape = mean_absolute_percentage_error(df_test['y'].values, forecast_test['yhat'].values) * 100
    n_cp = len(model.changepoints)

    print(f"{scale:<10} {rmse:<12.4f} {mape:<10.2f} {n_cp:<10}")
```

기대 결과 해석:
- scale=0.01: 변화점이 적고 추세가 매끄러움. 과소적합 가능
- scale=0.05: 기본값. 적절한 균형
- scale=0.5: 변화점이 많고 추세가 울퉁불퉁. 학습 데이터에는 잘 맞지만 테스트에서 불안정(과적합)
- 핵심: changepoint_prior_scale은 "추세의 유연성"을 조절하는 파라미터. 너무 높으면 학습 데이터의 노이즈까지 추세로 학습

### 프롬프트 4 모범 구현: Prophet 구성 요소별 기여도 분석

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from prophet import Prophet

data_path = Path(__file__).parent.parent / "data" / "co2.csv"
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
y = df['co2']

prophet_df = pd.DataFrame({'ds': y.index, 'y': y.values})
train_size = int(len(prophet_df) * 0.8)
df_train = prophet_df[:train_size]
df_test = prophet_df[train_size:]

model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, seasonality_mode='additive',
                changepoint_prior_scale=0.05)
model.fit(df_train)

future = model.make_future_dataframe(periods=len(df_test), freq='MS')
forecast = model.predict(future)
forecast_test = forecast[forecast['ds'] >= df_test['ds'].iloc[0]].copy()

# 각 성분의 절대값 비율 계산
forecast_test['trend_abs'] = forecast_test['trend'].abs()
forecast_test['yearly_abs'] = forecast_test['yearly'].abs()
forecast_test['total_abs'] = forecast_test['trend_abs'] + forecast_test['yearly_abs']
forecast_test['trend_ratio'] = forecast_test['trend_abs'] / forecast_test['total_abs'] * 100
forecast_test['yearly_ratio'] = forecast_test['yearly_abs'] / forecast_test['total_abs'] * 100

print("월별 구성 요소 기여도:")
print(f"{'날짜':<14} {'trend':<12} {'yearly':<12} {'trend%':<10} {'yearly%':<10}")
print("-" * 58)
for _, row in forecast_test.iterrows():
    print(f"{row['ds'].strftime('%Y-%m'):<14} {row['trend']:<12.2f} {row['yearly']:<12.2f} "
          f"{row['trend_ratio']:<10.1f} {row['yearly_ratio']:<10.1f}")

print(f"\n평균 trend 기여율: {forecast_test['trend_ratio'].mean():.1f}%")
print(f"평균 yearly 기여율: {forecast_test['yearly_ratio'].mean():.1f}%")
```

기대 결과:
- trend 기여율이 99% 이상으로 압도적. CO2 농도의 절대값(약 360-370ppm)이 크기 때문
- yearly 성분은 ±3ppm 정도로 절대 기여율은 1% 미만이지만, 예측 정확도에 핵심적 역할
- 핵심: "기여율이 크다"와 "예측에 중요하다"는 다른 개념. trend가 없으면 전혀 예측이 안 되고, yearly가 없으면 RMSE가 크게 증가한다

---

## 실습 3 해설: LSTM으로 CO2 농도 예측

### 제공 코드 실행 결과 해설

동일한 CO2 데이터에 2-layer LSTM(hidden_size=64)을 적용한 결과:

| 항목 | 값 경향 | 이유 |
| ---- | ------- | ---- |
| 총 파라미터 수 | 50,497개 | 2-layer LSTM(64) + FC(1) |
| RMSE | 7.53 ppm 전후 | 420개 샘플로 50K 파라미터 학습 부족 |
| MAPE | 1.83% 전후 | SARIMA(0.53%), Prophet(0.19%)보다 높음 |

핵심 코드 구조:

```python
# 시퀀스 생성: lookback 윈도우만큼의 과거 데이터를 입력으로 사용
# [t-24, t-23, ..., t-1] → t 예측
X_train, y_train = create_sequences(train_data, seq_length=24)

# MinMaxScaler: LSTM의 시그모이드/tanh 활성 함수에 맞게 0~1 범위로 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))
```

LSTM 성능이 낮은 핵심 이유: 420개 샘플로 50,497개 파라미터를 학습하는 것은 본질적으로 어렵다(비율 약 1:8). 일반적으로 파라미터 수의 10배 이상 샘플이 필요하다. CO2 데이터의 패턴(선형 추세 + 규칙적 계절성)은 LSTM의 비선형 표현력이 오히려 과도하다.

### 프롬프트 5 모범 구현: Lookback 윈도우 변경 실험

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

data_path = Path(__file__).parent.parent / "data" / "co2.csv"
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
y = df['co2']

scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

train_size = int(len(y_scaled) * 0.8)
train_data = y_scaled[:train_size]
test_data = y_scaled[train_size:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_lengths = [6, 12, 24, 36]
print(f"{'seq_length':<14} {'RMSE':<12} {'MAPE(%)':<10}")
print("-" * 36)

for seq_len in seq_lengths:
    torch.manual_seed(42)
    np.random.seed(42)

    X_train, y_train_seq = create_sequences(train_data, seq_len)
    X_test, y_test_seq = create_sequences(
        np.concatenate([train_data[-seq_len:], test_data]), seq_len
    )

    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train_seq)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

    model.train()
    for _ in range(100):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    preds = scaler.inverse_transform(preds)
    actuals = scaler.inverse_transform(y_test_seq)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mape = mean_absolute_percentage_error(actuals, preds) * 100
    print(f"{seq_len:<14} {rmse:<12.4f} {mape:<10.2f}")
```

기대 결과 해석:
- seq_length=6: 6개월 윈도우로는 연간 계절성(12개월 주기)을 포착할 수 없다. 성능이 가장 낮을 가능성
- seq_length=12: 1년 주기를 겨우 포함. 개선되지만 불충분
- seq_length=24: 2년 주기를 포함. 계절 패턴을 2회 관찰하여 안정적 학습
- seq_length=36: 더 많은 과거 정보를 보지만, 학습 시퀀스 수가 줄어들고 학습 난이도가 올라감
- 핵심: lookback 윈도우는 "최소 1-2 계절 주기"를 포함해야 하지만, 너무 길면 학습 데이터가 줄어들고 장기 의존성 학습이 어려워진다

### 프롬프트 6 모범 구현: hidden_size 변경 실험

```python
import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

data_path = Path(__file__).parent.parent / "data" / "co2.csv"
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
y = df['co2']

scaler = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

train_size = int(len(y_scaled) * 0.8)
train_data = y_scaled[:train_size]
test_data = y_scaled[train_size:]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 24
X_train, y_train_seq = create_sequences(train_data, seq_length)
X_test, y_test_seq = create_sequences(
    np.concatenate([train_data[-seq_length:], test_data]), seq_length
)

hidden_sizes = [16, 32, 64, 128]
print(f"{'hidden_size':<14} {'파라미터 수':<14} {'RMSE':<12} {'MAPE(%)':<10} {'시간(초)':<10}")
print("-" * 60)

for h_size in hidden_sizes:
    torch.manual_seed(42)
    np.random.seed(42)

    model = LSTMModel(hidden_size=h_size).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train_seq)
    loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=32, shuffle=True)

    start = time.time()
    model.train()
    for _ in range(100):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
    elapsed = time.time() - start

    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    preds = scaler.inverse_transform(preds)
    actuals = scaler.inverse_transform(y_test_seq)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mape = mean_absolute_percentage_error(actuals, preds) * 100
    print(f"{h_size:<14} {total_params:<14,} {rmse:<12.4f} {mape:<10.2f} {elapsed:<10.1f}")
```

기대 결과:
- hidden_size=16: 파라미터 수가 적어 학습 빠르지만 표현력 부족 가능
- hidden_size=64: 기본값. 적절한 균형
- hidden_size=128: 파라미터가 많아지지만 420개 샘플에서는 과적합 위험 증가
- 핵심: 이 데이터에서는 hidden_size를 늘려도 성능 개선이 제한적이다. 병목은 모형 복잡도가 아니라 데이터 양이기 때문

---

## 실습 4 해설: 세 모형 비교와 복잡 시계열 실험

### 제공 코드 실행 결과 해설

**CO2 데이터 비교 결과**:

| 모형 | RMSE 경향 | MAPE 경향 | 이유 |
| ---- | --------- | --------- | ---- |
| Prophet | 0.87 ppm | 0.19% | 추세+계절성 분리가 가장 효과적 |
| SARIMA | 2.30 ppm | 0.53% | 차분 기반 접근, 장기 예측에서 오차 누적 |
| LSTM | 7~12 ppm | 1.8~2.9% | 데이터 부족, 단순 패턴에 과도한 복잡성 |

**복잡 시계열 비교 결과** (합성 데이터, 5,000 샘플):

| 모형 | RMSE 경향 | 이유 |
| ---- | --------- | ---- |
| LSTM 다변량 | 27.27 | 비선형 외생 변수 효과를 암묵적 학습 |
| SARIMA | 29.29 | 자기상관 구조 포착, 비선형 효과 미포착 |
| Prophet | 42.90 | 외생 변수를 선형으로만 처리 |
| LSTM 단변량 | 100.17 | 외생 변수 없이 y만으로는 변동 설명 불가 |

핵심 교훈:
1. CO2(단순 패턴, 적은 데이터): Prophet > SARIMA > LSTM
2. 복잡 시계열(비선형, 많은 데이터, 외생 변수): LSTM 다변량 > SARIMA > Prophet > LSTM 단변량
3. "복잡한 모형이 항상 좋은 것은 아니다." 데이터 특성에 맞는 모형 선택이 핵심

### 프롬프트 7 모범 구현: SARIMA vs Prophet 장단기 예측 성능 비교

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from sklearn.metrics import mean_squared_error

data_path = Path(__file__).parent.parent / "data" / "co2.csv"
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
y = df['co2']

train_size = int(len(y) * 0.8)
y_train = y[:train_size]
y_test = y[train_size:]

# SARIMA 예측
sarima = SARIMAX(y_train, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12),
                 enforce_stationarity=False, enforce_invertibility=False)
sarima_result = sarima.fit(disp=False)
sarima_forecast = sarima_result.get_forecast(steps=len(y_test)).predicted_mean

# Prophet 예측
df_train = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                        daily_seasonality=False, seasonality_mode='additive')
prophet_model.fit(df_train)
future = prophet_model.make_future_dataframe(periods=len(y_test), freq='MS')
prophet_forecast = prophet_model.predict(future)
prophet_pred = prophet_forecast['yhat'].iloc[-len(y_test):].values

# 구간별 RMSE 비교
periods = [12, 24, 48, len(y_test)]
print(f"{'예측 기간':<14} {'SARIMA RMSE':<16} {'Prophet RMSE':<16} {'차이':<10}")
print("-" * 56)

for n in periods:
    actual = y_test.values[:n]
    rmse_s = np.sqrt(mean_squared_error(actual, sarima_forecast.values[:n]))
    rmse_p = np.sqrt(mean_squared_error(actual, prophet_pred[:n]))
    diff = rmse_s - rmse_p
    label = f"{n}개월" if n < len(y_test) else f"전체({n})"
    print(f"{label:<14} {rmse_s:<16.4f} {rmse_p:<16.4f} {diff:<10.4f}")
```

기대 결과:
- 12개월: 두 모형 모두 정확. 차이가 작을 수 있음
- 24~48개월: Prophet의 우위가 점차 뚜렷해짐. SARIMA는 신뢰구간이 넓어지며 오차 누적
- 전체(106개월): Prophet이 SARIMA 대비 약 60% 이상 낮은 RMSE
- 핵심: Prophet의 변화점 탐지 기능이 장기 추세 변화에 적응하므로, 장기 예측에서 SARIMA보다 안정적. SARIMA는 단기 예측에서 상대적으로 효율적

### 프롬프트 8 모범 구현: 주 단위 데이터에서 LSTM 성능 개선 확인

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

data_path = Path(__file__).parent.parent / "data" / "co2.csv"
df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def run_lstm(y_series, seq_length, label):
    torch.manual_seed(42)
    np.random.seed(42)

    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y_series.values.reshape(-1, 1))

    train_size = int(len(y_scaled) * 0.8)
    train_data = y_scaled[:train_size]
    test_data = y_scaled[train_size:]

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(
        np.concatenate([train_data[-seq_length:], test_data]), seq_length
    )

    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
        batch_size=32, shuffle=True
    )

    model.train()
    for _ in range(100):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    preds = scaler.inverse_transform(preds)
    actuals = scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mape = mean_absolute_percentage_error(actuals, preds) * 100
    return train_size, rmse, mape

# 월 단위
y_monthly = df['co2']
train_m, rmse_m, mape_m = run_lstm(y_monthly, seq_length=24, label="monthly")

# 주 단위
y_weekly = df['co2'].resample('W').mean().interpolate()
train_w, rmse_w, mape_w = run_lstm(y_weekly, seq_length=104, label="weekly")

print(f"{'데이터 주기':<14} {'학습 샘플':<14} {'RMSE (ppm)':<14} {'MAPE (%)':<10}")
print("-" * 52)
print(f"{'월 단위':<14} {train_m:<14} {rmse_m:<14.4f} {mape_m:<10.2f}")
print(f"{'주 단위':<14} {train_w:<14} {rmse_w:<14.4f} {mape_w:<10.2f}")

params = sum(p.numel() for p in LSTMModel().parameters())
print(f"\n총 파라미터 수: {params:,}")
print(f"월 단위 데이터/파라미터 비율: 1:{train_m/params:.1f}")
print(f"주 단위 데이터/파라미터 비율: 1:{train_w/params:.1f}")
```

기대 결과:
- 월 단위: RMSE 7~12 ppm, MAPE 1.8~2.9%. 학습 샘플 약 420개, 파라미터 대비 비율 약 1:8
- 주 단위: RMSE 2~3 ppm, MAPE 0.5~0.6%. 학습 샘플 약 1,800개, 파라미터 대비 비율 약 1:36
- 핵심: 동일한 LSTM 구조에서 데이터 양을 4배 이상 늘리면 RMSE가 약 3배 이상 개선된다. "파라미터 수 대비 데이터 비율"이 신경망 학습의 핵심 병목이며, 일반적으로 파라미터 수의 10배 이상 샘플이 필요하다. 이 실험은 "LSTM이 나쁜 것이 아니라 데이터가 부족했던 것"임을 확인시켜 준다

---

## 9장 전체 핵심 정리

```text
1. 시계열 분해: 추세+계절성+잔차. 가법 분해는 변동 크기가 일정할 때, 승법은 비례할 때.
   분해를 통해 데이터 구조를 이해하는 것이 모형 선택의 첫걸음이다.
2. 정상성과 차분: ADF 검정으로 확인, 차분으로 확보.
   CO2 데이터는 1차 차분만으로 정상성이 확보되었다 (p<0.001).
3. ARIMA/SARIMA: 자기상관 구조 기반. 모형 계수가 해석 가능하고 데이터가 적어도 안정적.
   CO2 데이터에서 RMSE 2.30ppm, MAPE 0.53%.
4. Prophet: 추세+계절성+휴일을 분리 모형화. 변화점 탐지로 장기 예측에 강함.
   CO2 데이터에서 RMSE 0.87ppm, MAPE 0.19%로 최고 성능.
5. LSTM: 게이트 메커니즘으로 장기 의존성 학습. 데이터 충분+비선형+다변량일 때 강함.
   CO2(420샘플)에서는 최하위, 복잡 시계열(4,000샘플)에서는 다변량으로 1위.
6. 모형 선택의 원칙: "복잡한 모형이 항상 좋은 것은 아니다."
   데이터가 단순하면 단순한 모형이, 복잡하면 복잡한 모형이 낫다.
   동일 데이터+동일 분할+동일 지표로 비교해야 공정한 평가가 가능하다.
```
