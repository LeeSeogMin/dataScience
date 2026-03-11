# 9-8-comparison.py
# ARIMA, Prophet, LSTM 모형 비교
# 동일한 데이터 분할로 세 모형의 예측 성능을 비교

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import argparse

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


class LSTMModel(nn.Module):
    """LSTM 시계열 예측 모형"""
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def load_data(freq: str):
    """CO2 데이터 로드"""
    data_path = Path(__file__).parent.parent / "data" / "co2.csv"
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')

    if freq == "monthly":
        y = df["co2"]  # 이미 월별 데이터
    elif freq == "weekly":
        y = df["co2"].resample("W").mean().interpolate()
    else:
        raise ValueError(f"지원하지 않는 freq: {freq}")

    return y


def output_suffix(freq: str) -> str:
    return "" if freq == "monthly" else f"_{freq}"


def parse_args():
    parser = argparse.ArgumentParser(description="SARIMA/Prophet/LSTM 예측 성능 비교")
    parser.add_argument("--freq", choices=["monthly", "weekly"], default="monthly")
    parser.add_argument("--lstm-epochs", type=int, default=100)
    parser.add_argument("--lstm-seq-length", type=int, default=None)
    return parser.parse_args()


def evaluate(actual, predicted):
    """RMSE, MAPE 계산"""
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return rmse, mape


def fit_sarima(y_train, y_test, seasonal_period: int):
    """SARIMA 모형 적합 및 예측"""
    model = SARIMAX(y_train, order=(0, 1, 1), seasonal_order=(0, 1, 1, seasonal_period),
                    enforce_stationarity=False, enforce_invertibility=False)    
    result = model.fit(disp=False)
    forecast = result.get_forecast(steps=len(y_test)).predicted_mean
    return forecast.values


def fit_prophet(y_train, y_test, prophet_freq: str):
    """Prophet 모형 적합 및 예측"""
    df_train = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})

    model = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, seasonality_mode='additive')       
    model.fit(df_train)

    future = model.make_future_dataframe(periods=len(y_test), freq=prophet_freq)
    forecast = model.predict(future)
    return forecast['yhat'].iloc[-len(y_test):].values


def fit_lstm(y_train, y_test, seq_length=24, epochs=100):
    """LSTM 모형 적합 및 예측"""
    # 데이터 정규화
    scaler = MinMaxScaler()
    y_all = np.concatenate([y_train.values, y_test.values])
    scaler.fit(y_train.values.reshape(-1, 1))

    train_scaled = scaler.transform(y_train.values.reshape(-1, 1))
    test_input = scaler.transform(y_all.reshape(-1, 1))

    # 시퀀스 생성
    def create_sequences(data, seq_len):
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i + seq_len])
            y.append(data[i + seq_len])
        return np.array(X), np.array(y)

    X_train, y_train_seq = create_sequences(train_scaled, seq_length)

    # 모형 학습
    model = LSTMModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train_seq)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for _ in range(epochs):
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

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


def main():
    args = parse_args()
    print("=" * 60)
    print("9-8 시계열 예측 모형 비교 실습")
    print("=" * 60)

    # 1. 데이터 로드 및 분할
    y = load_data(args.freq)
    seasonal_period = 12 if args.freq == "monthly" else 52
    prophet_freq = "MS" if args.freq == "monthly" else "W"

    if args.lstm_seq_length is not None:
        lstm_seq_length = args.lstm_seq_length
    else:
        lstm_seq_length = 24 if args.freq == "monthly" else 104

    suffix = output_suffix(args.freq)
    train_size = int(len(y) * 0.8)
    y_train = y[:train_size]
    y_test = y[train_size:]

    print(f"\n데이터 기간: {y.index[0]} ~ {y.index[-1]}")
    print(f"학습: {len(y_train)}개, 테스트: {len(y_test)}개")
    print(f"리샘플링 주기: {args.freq}, SARIMA 계절 주기: {seasonal_period}, LSTM lookback: {lstm_seq_length}")

    # 2. 각 모형 적합 및 예측
    print("\n" + "=" * 40)
    print("모형별 예측 수행")
    print("=" * 40)

    print("SARIMA 적합 중...")
    pred_sarima = fit_sarima(y_train, y_test, seasonal_period=seasonal_period)
    rmse_sarima, mape_sarima = evaluate(y_test.values, pred_sarima)
    print(f"  RMSE: {rmse_sarima:.4f}, MAPE: {mape_sarima:.2f}%")

    print("Prophet 적합 중...")
    pred_prophet = fit_prophet(y_train, y_test, prophet_freq=prophet_freq)
    rmse_prophet, mape_prophet = evaluate(y_test.values, pred_prophet)
    print(f"  RMSE: {rmse_prophet:.4f}, MAPE: {mape_prophet:.2f}%")

    print("LSTM 적합 중...")
    pred_lstm = fit_lstm(y_train, y_test, seq_length=lstm_seq_length, epochs=args.lstm_epochs)
    rmse_lstm, mape_lstm = evaluate(y_test.values, pred_lstm)
    print(f"  RMSE: {rmse_lstm:.4f}, MAPE: {mape_lstm:.2f}%")

    # 3. 결과 정리
    results = pd.DataFrame({
        '모형': [f'SARIMA(0,1,1)(0,1,1)_{seasonal_period}', 'Prophet (additive)', 'LSTM (2-layer)'],
        'RMSE (ppm)': [rmse_sarima, rmse_prophet, rmse_lstm],
        'MAPE (%)': [mape_sarima, mape_prophet, mape_lstm]
    })
    results = results.sort_values('RMSE (ppm)')

    print("\n" + "=" * 40)
    print("모형 비교 결과")
    print("=" * 40)
    print(results.to_string(index=False))

    # 4. 비교 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1) 전체 예측 비교
    ax = axes[0, 0]
    ax.plot(y_test.index, y_test.values, 'k-', label='실제값', linewidth=2)
    ax.plot(y_test.index, pred_sarima, 'b--', label='SARIMA', linewidth=1.5, alpha=0.8)
    ax.plot(y_test.index, pred_prophet, 'g--', label='Prophet', linewidth=1.5, alpha=0.8)
    ax.plot(y_test.index, pred_lstm, 'r--', label='LSTM', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('날짜')
    ax.set_ylabel('CO2 농도 (ppm)')
    ax.set_title('모형별 예측 비교')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # (2) 성능 지표 막대 그래프
    ax = axes[0, 1]
    models = ['SARIMA', 'Prophet', 'LSTM']
    rmse_values = [rmse_sarima, rmse_prophet, rmse_lstm]
    colors = ['blue', 'green', 'red']
    bars = ax.bar(models, rmse_values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE (ppm)')
    ax.set_title('모형별 RMSE 비교')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, rmse_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10)

    # (3) 예측 오차 시계열
    ax = axes[1, 0]
    ax.plot(y_test.index, y_test.values - pred_sarima, 'b-', label='SARIMA', alpha=0.7)
    ax.plot(y_test.index, y_test.values - pred_prophet, 'g-', label='Prophet', alpha=0.7)
    ax.plot(y_test.index, y_test.values - pred_lstm, 'r-', label='LSTM', alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax.set_xlabel('날짜')
    ax.set_ylabel('오차 (ppm)')
    ax.set_title('모형별 예측 오차')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    # (4) 오차 분포 박스플롯
    ax = axes[1, 1]
    error_data = [y_test.values - pred_sarima,
                  y_test.values - pred_prophet,
                  y_test.values - pred_lstm]
    bp = ax.boxplot(error_data, labels=models, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    ax.set_ylabel('오차 (ppm)')
    ax.set_title('모형별 오차 분포')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"9-8-model-comparison{suffix}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n비교 그래프 저장: {output_dir / f'9-8-model-comparison{suffix}.png'}")

    # 5. 결과 저장
    results_dict = {
        "SARIMA": {"rmse": round(rmse_sarima, 4), "mape": round(mape_sarima, 2)},
        "Prophet": {"rmse": round(rmse_prophet, 4), "mape": round(mape_prophet, 2)},
        "LSTM": {"rmse": round(rmse_lstm, 4), "mape": round(mape_lstm, 2)},
        "best_model": results.iloc[0]['모형'],
        "data": {"frequency": args.freq},
        "train_size": len(y_train),
        "test_size": len(y_test)
    }

    with open(output_dir / f"ch9_model_comparison{suffix}.json", "w") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {output_dir / f'ch9_model_comparison{suffix}.json'}")

    # 예측값 CSV
    forecast_df = pd.DataFrame({
        'date': y_test.index,
        'actual': y_test.values,
        'sarima': pred_sarima,
        'prophet': pred_prophet,
        'lstm': pred_lstm
    })
    forecast_df.to_csv(output_dir / f"ch9_forecasts{suffix}.csv", index=False)
    print(f"예측값 저장: {output_dir / f'ch9_forecasts{suffix}.csv'}")

    return results_dict


if __name__ == "__main__":
    main()
