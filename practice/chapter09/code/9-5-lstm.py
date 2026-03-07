# 9-5-lstm.py
# LSTM을 이용한 CO2 농도 시계열 예측
# PyTorch 기반 순환신경망 구현

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

# 디바이스 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LSTMModel(nn.Module):
    """LSTM 시계열 예측 모형"""

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # LSTM 출력
        lstm_out, _ = self.lstm(x)
        # 마지막 시점의 출력만 사용
        out = self.fc(lstm_out[:, -1, :])
        return out


def create_sequences(data, seq_length):
    """시퀀스 데이터 생성 (lookback window)"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)


def train_model(model, train_loader, criterion, optimizer, num_epochs, verbose=True):
    """모형 학습"""
    model.train()
    train_losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")

    return train_losses


def evaluate_model(model, X_test, y_test, scaler):
    """모형 평가"""
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        predictions = model(X_test_tensor).cpu().numpy()

    # 역변환 (원래 스케일로)
    predictions = scaler.inverse_transform(predictions)
    actuals = scaler.inverse_transform(y_test)

    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mape = mean_absolute_percentage_error(actuals, predictions) * 100

    return predictions, actuals, rmse, mape


def load_co2_series(freq: str) -> pd.Series:
    """CO2 데이터를 지정한 주기로 리샘플링하여 반환"""
    data_path = Path(__file__).parent.parent / "data" / "co2.csv"
    df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')

    if freq == "monthly":
        series = df["co2"]  # 이미 월별 데이터
    elif freq == "weekly":
        series = df["co2"].resample("W").mean().interpolate()
    else:
        raise ValueError(f"지원하지 않는 freq: {freq}")

    return series


def output_suffix(freq: str) -> str:
    return "" if freq == "monthly" else f"-{freq}"


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM으로 CO2 시계열 예측 실습")
    parser.add_argument("--freq", choices=["monthly", "weekly"], default="monthly")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seq-length", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("9-5 LSTM 시계열 예측 실습")
    print("=" * 60)
    print(f"디바이스: {device}")
    print(f"리샘플링 주기: {args.freq}")

    # 1. 데이터 로드
    y = load_co2_series(args.freq)

    print(f"\n데이터 기간: {y.index[0]} ~ {y.index[-1]}")
    print(f"총 관측치 수: {len(y)}")

    # 2. 데이터 전처리 (정규화)
    scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

    # 3. 학습/테스트 분할
    train_size = int(len(y_scaled) * 0.8)
    train_data = y_scaled[:train_size]
    test_data = y_scaled[train_size:]

    print(f"\n학습 데이터: {train_size}개")
    print(f"테스트 데이터: {len(test_data)}개")

    # 4. 시퀀스 생성
    if args.seq_length is not None:
        seq_length = args.seq_length
    else:
        seq_length = 24 if args.freq == "monthly" else 104

    window_unit = "개월" if args.freq == "monthly" else "주"
    print(f"\nLookback 윈도우: {seq_length}{window_unit}")

    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(
        np.concatenate([train_data[-seq_length:], test_data]),
        seq_length
    )

    print(f"학습 시퀀스: {X_train.shape}")
    print(f"테스트 시퀀스: {X_test.shape}")

    # 5. 텐서 변환 및 데이터로더
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # 6. 모형 초기화
    model = LSTMModel(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    print("\n" + "=" * 40)
    print("LSTM 모형 구조")
    print("=" * 40)
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n총 파라미터 수: {total_params:,}")

    # 7. 학습 설정
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = args.epochs

    print("\n" + "=" * 40)
    print("모형 학습")
    print("=" * 40)

    train_losses = train_model(
        model, train_loader, criterion, optimizer, num_epochs
    )

    # 8. 평가
    print("\n" + "=" * 40)
    print("예측 성능")
    print("=" * 40)

    predictions, actuals, rmse, mape = evaluate_model(model, X_test, y_test, scaler)

    print(f"RMSE: {rmse:.4f} ppm")
    print(f"MAPE: {mape:.2f}%")

    suffix = output_suffix(args.freq)

    # 9. 학습 곡선 시각화
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(train_losses, 'b-', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('LSTM 학습 곡선')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f"9-5-lstm{suffix}-loss.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n학습 곡선 저장: {output_dir / f'9-5-lstm{suffix}-loss.png'}")

    # 10. 예측 결과 시각화
    # 테스트 시퀀스 수에 맞춰 날짜 생성
    test_dates = y.index[train_size:train_size + len(predictions)]

    fig, ax = plt.subplots(figsize=(14, 6))

    # 전체 데이터
    ax.plot(y.index[:train_size], y.values[:train_size], 'b-',
            label='학습 데이터', linewidth=1)

    # 테스트 실제값
    ax.plot(test_dates, actuals.flatten(), 'g-',
            label='실제값 (테스트)', linewidth=1.5)

    # 예측값
    ax.plot(test_dates, predictions.flatten(), 'r--',
            label='LSTM 예측', linewidth=1.5)

    ax.set_xlabel('날짜')
    ax.set_ylabel('CO2 농도 (ppm)')
    ax.set_title(f'LSTM 예측 결과 (RMSE={rmse:.2f}, MAPE={mape:.2f}%)')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"9-5-lstm{suffix}-forecast.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"예측 그래프 저장: {output_dir / f'9-5-lstm{suffix}-forecast.png'}")

    # 11. 예측 오차 분석
    errors = actuals.flatten() - predictions.flatten()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 오차 시계열
    axes[0].plot(test_dates, errors, 'purple', linewidth=1)
    axes[0].axhline(y=0, color='k', linestyle='--', linewidth=0.8)
    axes[0].set_xlabel('날짜')
    axes[0].set_ylabel('오차 (ppm)')
    axes[0].set_title('예측 오차 시계열')
    axes[0].grid(True, alpha=0.3)

    # 오차 히스토그램
    axes[1].hist(errors, bins=20, color='purple', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=0, color='k', linestyle='--', linewidth=0.8)
    axes[1].set_xlabel('오차 (ppm)')
    axes[1].set_ylabel('빈도')
    axes[1].set_title(f'오차 분포 (평균: {errors.mean():.2f}, 표준편차: {errors.std():.2f})')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"9-5-lstm{suffix}-errors.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"오차 분석 그래프 저장: {output_dir / f'9-5-lstm{suffix}-errors.png'}")

    # 12. 결과 저장
    results = {
        "model": "LSTM",
        "data": {
            "frequency": args.freq,
            "n_observations": int(len(y)),
            "train_size": int(train_size),
            "test_size": int(len(test_data)),
        },
        "architecture": {
            "input_size": 1,
            "hidden_size": 64,
            "num_layers": 2,
            "seq_length": seq_length,
            "total_params": total_params
        },
        "training": {
            "epochs": num_epochs,
            "batch_size": 32,
            "learning_rate": 0.001,
            "final_loss": round(train_losses[-1], 6)
        },
        "performance": {
            "rmse": round(rmse, 4),
            "mape": round(mape, 2)
        },
        "error_stats": {
            "mean": round(float(errors.mean()), 4),
            "std": round(float(errors.std()), 4),
            "min": round(float(errors.min()), 4),
            "max": round(float(errors.max()), 4)
        }
    }

    with open(output_dir / f"9-5-lstm{suffix}-results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"결과 저장: {output_dir / f'9-5-lstm{suffix}-results.json'}")

    # 예측값 CSV 저장
    forecast_df = pd.DataFrame({
        'date': test_dates,
        'actual': actuals.flatten(),
        'forecast': predictions.flatten(),
        'error': errors
    })
    forecast_df.to_csv(output_dir / f"9-5-lstm{suffix}-forecast.csv", index=False)
    print(f"예측값 저장: {output_dir / f'9-5-lstm{suffix}-forecast.csv'}")

    return results


if __name__ == "__main__":
    main()
