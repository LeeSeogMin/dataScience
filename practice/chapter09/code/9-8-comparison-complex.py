# 9-8-comparison-complex.py
# 대규모 복잡 시계열에서의 ARIMA, Prophet, LSTM 성능 비교
#
# 목적: LSTM이 강점을 발휘할 수 있는 조건에서의 성능 검증
# - 충분한 샘플 수 (5,000개 이상)
# - 비선형 패턴 포함
# - 다변량 상호작용 (외생 변수 영향)
# - 복잡한 계절성 (다중 주기)

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

# 재현성을 위한 시드 설정
np.random.seed(42)
torch.manual_seed(42)

# 출력 디렉토리 설정
output_dir = Path(__file__).parent.parent / "data"
diagram_dir = Path(__file__).parent.parent.parent.parent / "diagram"

# 한글 폰트 설정 (크로스 플랫폼)
import platform
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':  # macOS
    plt.rc('font', family='AppleGothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')
plt.rc('axes', unicode_minus=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_complex_timeseries():
    """복잡한 시계열 데이터 로드"""
    data_path = Path(__file__).parent.parent / "data" / "complex_timeseries.csv"
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
    return df


class LSTMModel(nn.Module):
    """향상된 LSTM 모형 (더 큰 용량)"""
    def __init__(self, input_size=1, hidden_size=128, num_layers=3, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.relu(self.fc1(lstm_out[:, -1, :]))
        return self.fc2(out)


class LSTMMultivariate(nn.Module):
    """다변량 LSTM 모형 (외생 변수 포함)"""
    def __init__(self, input_size=3, hidden_size=128, num_layers=3, dropout=0.2):
        super(LSTMMultivariate, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.relu(self.fc1(lstm_out[:, -1, :]))
        return self.fc2(out)


def evaluate(actual, predicted):
    """RMSE, MAPE 계산"""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return rmse, mape


def fit_sarima(y_train, y_test):
    """SARIMA 모형 적합 및 예측"""
    # 일간 데이터에서 주간 계절성 (7일)
    model = SARIMAX(y_train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False, maxiter=100)
    forecast = result.get_forecast(steps=len(y_test)).predicted_mean
    return forecast.values


def fit_prophet(y_train, y_test, df_full):
    """Prophet 모형 적합 (다중 계절성 + 외생 변수)"""
    # 학습 데이터 준비
    train_temp = df_full.loc[y_train.index, 'temperature'].values
    df_prophet = pd.DataFrame({
        'ds': y_train.index,
        'y': y_train.values,
        'temperature': train_temp
    })

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        changepoint_prior_scale=0.1
    )
    model.add_regressor('temperature')
    model.fit(df_prophet)

    # 테스트 데이터 준비
    test_temp = df_full.loc[y_test.index, 'temperature'].values
    future = pd.DataFrame({
        'ds': y_test.index,
        'temperature': test_temp
    })

    # 예측
    forecast = model.predict(future)
    return forecast['yhat'].values


def fit_lstm_univariate(y_train, y_test, seq_length=60, epochs=50, batch_size=64):
    """단변량 LSTM 모형"""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(y_train.values.reshape(-1, 1))

    # 시퀀스 생성
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    X_train, y_train_seq = create_sequences(train_scaled, seq_length)

    # 모형 학습
    model = LSTMModel(input_size=1, hidden_size=128, num_layers=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train_seq)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(loader))

    # 예측 (롤링 방식)
    model.eval()
    predictions = []
    current_seq = train_scaled[-seq_length:].copy()

    with torch.no_grad():
        for _ in range(len(y_test)):
            x = torch.FloatTensor(current_seq.reshape(1, seq_length, 1)).to(device)
            pred = model(x).cpu().numpy()[0, 0]
            predictions.append(pred)
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions


def fit_lstm_multivariate(df_train, df_test, seq_length=60, epochs=50, batch_size=64):
    """다변량 LSTM 모형 (복수 외생 변수 포함)"""
    # 스케일러 준비
    scaler_y = MinMaxScaler()
    scaler_temp = MinMaxScaler()
    scaler_econ = MinMaxScaler()

    y_scaled = scaler_y.fit_transform(df_train['y'].values.reshape(-1, 1))
    temp_scaled = scaler_temp.fit_transform(df_train['temperature'].values.reshape(-1, 1))
    econ_scaled = scaler_econ.fit_transform(df_train['economic_idx'].values.reshape(-1, 1))

    train_data = np.column_stack([y_scaled, temp_scaled, econ_scaled])

    # 시퀀스 생성
    def create_sequences_multi(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len, 0])  # y만 예측
        return np.array(X), np.array(y)

    X_train, y_train_seq = create_sequences_multi(train_data, seq_length)

    # 모형 학습
    model = LSTMMultivariate(input_size=3, hidden_size=128, num_layers=3).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train_seq.reshape(-1, 1))
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step(epoch_loss / len(loader))

    # 예측 - 테스트 기간의 실제 외생 변수 사용
    model.eval()
    predictions = []

    # 초기 시퀀스
    current_y = y_scaled[-seq_length:].copy()
    current_temp = temp_scaled[-seq_length:].copy()
    current_econ = econ_scaled[-seq_length:].copy()

    # 테스트 기간 외생 변수 스케일링
    test_temp_scaled = scaler_temp.transform(df_test['temperature'].values.reshape(-1, 1))
    test_econ_scaled = scaler_econ.transform(df_test['economic_idx'].values.reshape(-1, 1))

    with torch.no_grad():
        for i in range(len(df_test)):
            current_seq = np.column_stack([current_y, current_temp, current_econ])
            x = torch.FloatTensor(current_seq.reshape(1, seq_length, 3)).to(device)
            pred = model(x).cpu().numpy()[0, 0]
            predictions.append(pred)

            # 시퀀스 업데이트
            current_y = np.roll(current_y, -1)
            current_y[-1] = pred
            current_temp = np.roll(current_temp, -1)
            current_temp[-1] = test_temp_scaled[i]
            current_econ = np.roll(current_econ, -1)
            current_econ[-1] = test_econ_scaled[i]

    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
    return predictions


def main():
    print("=" * 70)
    print("9-8 대규모 복잡 시계열에서의 모형 비교 실험")
    print("=" * 70)

    # 1. 복잡한 시계열 데이터 로드
    print("\n1. 복잡한 시계열 데이터 로드 중...")
    df = load_complex_timeseries()

    print(f"   - 총 샘플 수: {len(df)}")
    print(f"   - 기간: {df.index[0].date()} ~ {df.index[-1].date()}")
    print(f"   - 특징: 비선형 추세, 다중 계절성, 복수 외생 변수(온도+경제지표), 이분산성, 비선형 자기상관")

    # 2. 학습/테스트 분할 (80:20)
    train_size = int(len(df) * 0.8)
    df_train = df.iloc[:train_size]
    df_test = df.iloc[train_size:]

    y_train = df_train['y']
    y_test = df_test['y']

    print(f"\n2. 데이터 분할")
    print(f"   - 학습: {len(y_train)}개 ({df_train.index[0].date()} ~ {df_train.index[-1].date()})")
    print(f"   - 테스트: {len(y_test)}개 ({df_test.index[0].date()} ~ {df_test.index[-1].date()})")

    # 3. 모형 비교
    print("\n3. 모형별 예측 수행")
    print("-" * 50)

    # SARIMA
    print("   SARIMA(2,1,2)(1,1,1)₇ 적합 중...")
    try:
        pred_sarima = fit_sarima(y_train, y_test)
        rmse_sarima, mape_sarima = evaluate(y_test.values, pred_sarima)
        print(f"   → RMSE: {rmse_sarima:.4f}, MAPE: {mape_sarima:.2f}%")
    except Exception as e:
        print(f"   → SARIMA 실패: {e}")
        pred_sarima = np.full(len(y_test), y_train.mean())
        rmse_sarima, mape_sarima = evaluate(y_test.values, pred_sarima)

    # Prophet
    print("\n   Prophet (다중 계절성 + 외생 변수) 적합 중...")
    pred_prophet = fit_prophet(y_train, y_test, df)
    rmse_prophet, mape_prophet = evaluate(y_test.values, pred_prophet)
    print(f"   → RMSE: {rmse_prophet:.4f}, MAPE: {mape_prophet:.2f}%")

    # LSTM (단변량) - 더 긴 시퀀스, 더 많은 에폭
    print("\n   LSTM 단변량 (3-layer, hidden=128, seq=90) 적합 중...")
    pred_lstm_uni = fit_lstm_univariate(y_train, y_test, seq_length=90, epochs=100)
    rmse_lstm_uni, mape_lstm_uni = evaluate(y_test.values, pred_lstm_uni)
    print(f"   → RMSE: {rmse_lstm_uni:.4f}, MAPE: {mape_lstm_uni:.2f}%")

    # LSTM (다변량) - 외생 변수 2개 (온도 + 경제 지표)
    print("\n   LSTM 다변량 (3-layer, hidden=128, +온도+경제지표, seq=90) 적합 중...")
    pred_lstm_multi = fit_lstm_multivariate(df_train, df_test, seq_length=90, epochs=100)
    rmse_lstm_multi, mape_lstm_multi = evaluate(y_test.values, pred_lstm_multi)
    print(f"   → RMSE: {rmse_lstm_multi:.4f}, MAPE: {mape_lstm_multi:.2f}%")

    # 4. 결과 정리
    print("\n" + "=" * 70)
    print("모형 비교 결과 (대규모 복잡 시계열)")
    print("=" * 70)

    results = pd.DataFrame({
        '모형': ['SARIMA(2,1,2)(1,1,1)₇', 'Prophet (다중계절성+외생)',
                'LSTM 단변량', 'LSTM 다변량'],
        'RMSE': [rmse_sarima, rmse_prophet, rmse_lstm_uni, rmse_lstm_multi],
        'MAPE (%)': [mape_sarima, mape_prophet, mape_lstm_uni, mape_lstm_multi]
    })
    results = results.sort_values('RMSE')
    results['순위'] = range(1, len(results) + 1)

    print(results.to_string(index=False))

    # 5. 시각화
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (1) 전체 예측 비교 (일부 구간)
    ax = axes[0, 0]
    plot_range = slice(0, min(200, len(y_test)))  # 처음 200일
    ax.plot(y_test.index[plot_range], y_test.values[plot_range], 'k-',
            label='실제값', linewidth=2)
    ax.plot(y_test.index[plot_range], pred_sarima[plot_range], 'b--',
            label='SARIMA', linewidth=1.5, alpha=0.8)
    ax.plot(y_test.index[plot_range], pred_prophet[plot_range], 'g--',
            label='Prophet', linewidth=1.5, alpha=0.8)
    ax.plot(y_test.index[plot_range], pred_lstm_multi[plot_range], 'r--',
            label='LSTM 다변량', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('날짜')
    ax.set_ylabel('값')
    ax.set_title('모형별 예측 비교 (테스트 초반 200일)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # (2) RMSE 비교
    ax = axes[0, 1]
    models = ['SARIMA', 'Prophet', 'LSTM\n(단변량)', 'LSTM\n(다변량)']
    rmse_values = [rmse_sarima, rmse_prophet, rmse_lstm_uni, rmse_lstm_multi]
    colors = ['blue', 'green', 'orange', 'red']
    bars = ax.bar(models, rmse_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE')
    ax.set_title('모형별 RMSE 비교')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # (3) 예측 오차 시계열 (처음 200일)
    ax = axes[1, 0]
    ax.plot(y_test.index[plot_range], y_test.values[plot_range] - pred_sarima[plot_range],
            'b-', label='SARIMA', alpha=0.7)
    ax.plot(y_test.index[plot_range], y_test.values[plot_range] - pred_prophet[plot_range],
            'g-', label='Prophet', alpha=0.7)
    ax.plot(y_test.index[plot_range], y_test.values[plot_range] - pred_lstm_multi[plot_range],
            'r-', label='LSTM 다변량', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('날짜')
    ax.set_ylabel('오차')
    ax.set_title('모형별 예측 오차 (테스트 초반 200일)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # (4) 오차 분포 박스플롯
    ax = axes[1, 1]
    error_data = [
        y_test.values - pred_sarima,
        y_test.values - pred_prophet,
        y_test.values - pred_lstm_uni,
        y_test.values - pred_lstm_multi
    ]
    bp = ax.boxplot(error_data, labels=['SARIMA', 'Prophet', 'LSTM\n(단변량)', 'LSTM\n(다변량)'],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax.set_ylabel('오차')
    ax.set_title('모형별 오차 분포')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "9-8-complex-comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n비교 그래프 저장: {output_dir / '9-8-complex-comparison.png'}")

    # 6. 결과 저장
    results_dict = {
        "experiment": "대규모 복잡 시계열 비교",
        "data": {
            "n_samples": n_samples,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "features": "비선형 추세, 다중 계절성(주간+연간), 외생변수(온도), 이분산성"
        },
        "results": {
            "SARIMA": {"rmse": round(rmse_sarima, 4), "mape": round(mape_sarima, 2)},
            "Prophet": {"rmse": round(rmse_prophet, 4), "mape": round(mape_prophet, 2)},
            "LSTM_univariate": {"rmse": round(rmse_lstm_uni, 4), "mape": round(mape_lstm_uni, 2)},
            "LSTM_multivariate": {"rmse": round(rmse_lstm_multi, 4), "mape": round(mape_lstm_multi, 2)}
        },
        "best_model": results.iloc[0]['모형'],
        "conclusion": "대규모 복잡 시계열에서 LSTM의 성능 검증"
    }

    with open(output_dir / "ch9_complex_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {output_dir / 'ch9_complex_comparison.json'}")

    # 7. 분석 요약
    print("\n" + "=" * 70)
    print("분석 요약")
    print("=" * 70)
    print(f"\n최고 성능 모형: {results.iloc[0]['모형']}")
    print(f"  - RMSE: {results.iloc[0]['RMSE']:.4f}")
    print(f"  - MAPE: {results.iloc[0]['MAPE (%)']:.2f}%")

    # LSTM vs 기존 모형 비교
    lstm_best = min(rmse_lstm_uni, rmse_lstm_multi)
    traditional_best = min(rmse_sarima, rmse_prophet)

    if lstm_best < traditional_best:
        improvement = (traditional_best - lstm_best) / traditional_best * 100
        print(f"\n→ LSTM이 기존 모형 대비 {improvement:.1f}% 더 낮은 RMSE 달성")
    else:
        print(f"\n→ 기존 모형이 여전히 우수 (LSTM RMSE: {lstm_best:.2f}, 기존 최고: {traditional_best:.2f})")

    return results_dict


if __name__ == "__main__":
    results = main()
