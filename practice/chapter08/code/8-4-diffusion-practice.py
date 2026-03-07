"""
8.4 Diffusion Model 불확실성 정량화 실습

목표:
- Diffusion Model로 시계열 미래 분포 생성
- 다중 시나리오 샘플링
- 불확실성 시각화 (분위수, 팬 차트)

데이터: 월별 매출 시계열 (practice/chapter8/data/input/monthly_sales.csv)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

def configure_matplotlib(output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / "mplconfig"))
    (output_dir / "mplconfig").mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

# 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "input"
OUTPUT_DIR = DATA_DIR / "output" / "8-4-diffusion-practice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class SimpleDiffusion(nn.Module):
    """간단한 조건부 Diffusion 모델"""
    def __init__(self, input_dim=1, condition_dim=12, hidden_dim=64):
        super().__init__()
        self.T = 50  # Diffusion steps (실습용으로 축소)

        # Noise prediction network (조건부)
        self.net = nn.Sequential(
            nn.Linear(input_dim + condition_dim + 1, hidden_dim),  # +1 for timestep
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def add_noise(self, x, t):
        """Forward diffusion: 점진적으로 노이즈 추가"""
        # 선형 스케줄
        beta_t = torch.linspace(1e-4, 0.02, self.T)[t]
        alpha_t = 1 - beta_t
        alpha_bar_t = torch.cumprod(alpha_t, dim=0)

        noise = torch.randn_like(x)
        noisy_x = torch.sqrt(alpha_bar_t.unsqueeze(-1)) * x + torch.sqrt(1 - alpha_bar_t.unsqueeze(-1)) * noise

        return noisy_x, noise

    def predict_noise(self, x_t, t, condition):
        """노이즈 예측 (조건부)"""
        # Timestep 임베딩
        t_embed = t.float().unsqueeze(-1) / self.T

        # 조건, 노이즈 데이터, 타임스텝 결합
        input_tensor = torch.cat([x_t, condition, t_embed], dim=-1)
        return self.net(input_tensor)

    def reverse_diffusion(self, condition, n_samples=100):
        """Reverse diffusion: 노이즈에서 데이터 생성"""
        # 순수 노이즈로 시작
        x_t = torch.randn(n_samples, 1)

        # 조건 복제
        condition = condition.unsqueeze(0).repeat(n_samples, 1)

        # 역방향 과정
        for t in reversed(range(self.T)):
            t_tensor = torch.tensor([t] * n_samples)
            noise_pred = self.predict_noise(x_t, t_tensor, condition)

            # 간단한 업데이트 (DDPM 공식 간소화)
            beta_t = torch.linspace(1e-4, 0.02, self.T)[t]
            x_t = x_t - beta_t * noise_pred

            if t > 0:
                x_t = x_t + torch.sqrt(beta_t) * torch.randn_like(x_t)

        return x_t.detach().numpy().flatten()

def generate_sales_data(n_months=60):
    """월별 매출 시계열 로드"""
    print("시계열 데이터 로드 중...")
    df = pd.read_csv(INPUT_DIR / "monthly_sales.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"데이터 로드 완료: {len(df)}개월")
    print(f"매출 범위: {df['sales'].min():.2f} ~ {df['sales'].max():.2f}")
    return df

def create_sequences(data, lookback=12):
    """시계열 시퀀스 생성 (과거 12개월 → 다음 1개월 예측)"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

def train_diffusion(X_train, y_train, epochs=100):
    """Diffusion 모델 학습"""
    print("\n" + "="*60)
    print("Diffusion 모델 학습")
    print("="*60)

    device = torch.device('cpu')
    model = SimpleDiffusion(input_dim=1, condition_dim=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 데이터 준비
    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(-1)

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 학습
    print(f"학습 시작 (Epochs={epochs})...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        for batch_cond, batch_y in loader:
            batch_cond = batch_cond.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # 랜덤 타임스텝 샘플링
            t = torch.randint(0, model.T, (len(batch_y),))

            # 노이즈 추가
            noisy_y, true_noise = model.add_noise(batch_y, t)

            # 노이즈 예측
            pred_noise = model.predict_noise(noisy_y, t, batch_cond)

            # 손실 계산
            loss = nn.MSELoss()(pred_noise, true_noise)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("학습 완료")
    return model

def forecast_with_uncertainty(model, last_sequence, n_samples=100):
    """불확실성 포함 예측"""
    print(f"\n{n_samples}개의 미래 경로 샘플링...")

    model.eval()
    condition = torch.FloatTensor(last_sequence)

    # 다중 샘플 생성
    samples = model.reverse_diffusion(condition, n_samples=n_samples)

    # 분위수 계산
    p5 = np.percentile(samples, 5)
    p25 = np.percentile(samples, 25)
    p50 = np.percentile(samples, 50)
    p75 = np.percentile(samples, 75)
    p95 = np.percentile(samples, 95)

    print(f"예측 분포 (스케일된 값):")
    print(f"  5%ile (최악): {p5:.2f}")
    print(f"  25%ile: {p25:.2f}")
    print(f"  중앙값: {p50:.2f}")
    print(f"  75%ile: {p75:.2f}")
    print(f"  95%ile (낙관): {p95:.2f}")

    return {
        'samples': samples,
        'p5': p5,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'p95': p95
    }

def visualize_forecast(df, forecast_results, n_history=24):
    """예측 불확실성 시각화 (흑백)"""
    print("\n예측 시각화 생성 중...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 1. 시계열 + 팬 차트
    ax1 = axes[0]

    # 과거 데이터 (검정색 실선)
    history = df.tail(n_history)
    ax1.plot(range(len(history)), history['sales'], 'o-', color='black', label='Actual', linewidth=2)

    # 미래 예측
    future_idx = len(history)
    ax1.axvline(future_idx - 1, color='gray', linestyle='--', alpha=0.7, label='Forecast start')

    # 불확실성 팬 차트
    p5 = forecast_results['p5']
    p25 = forecast_results['p25']
    p50 = forecast_results['p50']
    p75 = forecast_results['p75']
    p95 = forecast_results['p95']

    # 중앙값 예측 (검정색 사각 마커)
    ax1.plot(
        [future_idx - 1, future_idx],
        [history['sales'].iloc[-1], p50],
        's-',
        color='black',
        linewidth=2,
        markersize=8,
        label='Median forecast',
    )

    # 90% 예측구간 (5%ile ~ 95%ile) - 연한 회색
    ax1.fill_between([future_idx - 1, future_idx],
                     [history['sales'].iloc[-1], p5],
                     [history['sales'].iloc[-1], p95],
                     color='gray', alpha=0.3, label='90% PI')

    # 50% 예측구간 (25%ile ~ 75%ile) - 진한 회색
    ax1.fill_between([future_idx - 1, future_idx],
                     [history['sales'].iloc[-1], p25],
                     [history['sales'].iloc[-1], p75],
                     color='gray', alpha=0.5, label='50% PI')

    ax1.set_xlabel('Time (Months)')
    ax1.set_ylabel('Sales')
    ax1.set_title('Diffusion forecast uncertainty (fan chart)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 예측 분포 히스토그램 (흑백)
    ax2 = axes[1]

    samples = forecast_results['samples']
    ax2.hist(samples, bins=50, color='lightgray', edgecolor='black', alpha=0.8, density=True)

    # 분위수 선 추가 (흑백 - 선 스타일로 구분)
    ax2.axvline(p5, color='black', linestyle=':', linewidth=2, label=f'5%ile: {p5:.2f}')
    ax2.axvline(p50, color='black', linestyle='-', linewidth=2.5, label=f'median: {p50:.2f}')
    ax2.axvline(p95, color='black', linestyle='--', linewidth=2, label=f'95%ile: {p95:.2f}')

    ax2.set_xlabel('Predicted Sales')
    ax2.set_ylabel('Density')
    ax2.set_title('Forecast distribution (samples)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "diffusion_forecast_uncertainty.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"저장 완료: {save_path}")
    plt.close()

def create_risk_analysis(forecast_results):
    """리스크 분석표 생성"""
    print("\n" + "="*60)
    print("리스크 시나리오 분석")
    print("="*60)

    p5 = forecast_results['p5']
    p25 = forecast_results['p25']
    p50 = forecast_results['p50']
    p75 = forecast_results['p75']
    p95 = forecast_results['p95']

    risk_table = pd.DataFrame({
        '시나리오': ['최악 (5%ile)', '보수적 (25%ile)', '중립 (중앙값)', '낙관적 (75%ile)', '최선 (95%ile)'],
        '예측 매출': [f"{p5:.2f}", f"{p25:.2f}", f"{p50:.2f}", f"{p75:.2f}", f"{p95:.2f}"],
        '발생 확률': ['5%', '25%', '50%', '75%', '95%'],
        '리스크 대응': [
            '긴급 비용 절감',
            '보수적 예산',
            '계획대로 진행',
            '공격적 투자',
            '확장 계획'
        ]
    })

    print("\n" + risk_table.to_string(index=False))

    # CSV 저장
    risk_table.to_csv(OUTPUT_DIR / "diffusion_risk_scenarios.csv", index=False, encoding='utf-8-sig')
    print(f"\n리스크 시나리오 저장: {OUTPUT_DIR / 'diffusion_risk_scenarios.csv'}")

    # 불확실성 범위 계산
    uncertainty_range = p95 - p5
    median = p50

    print(f"\n[불확실성 지표]:")
    print(f"중앙값: {median:.2f}")
    print(f"불확실성 범위: {uncertainty_range:.2f} (95%ile - 5%ile)")
    print(f"상대적 불확실성: {(uncertainty_range/median)*100:.1f}%")

    if (uncertainty_range / median) < 0.2:
        print("✅ 예측 불확실성 낮음 (안정적)")
    elif (uncertainty_range / median) < 0.4:
        print("⚠️ 예측 불확실성 중간 (주의 필요)")
    else:
        print("⚠️ 예측 불확실성 높음 (리스크 관리 강화)")

    return risk_table

def main():
    print("="*60)
    print("8.4 Diffusion Model 불확실성 정량화 실습")
    print("="*60)

    set_seed(42)
    configure_matplotlib(OUTPUT_DIR)

    # 1. 데이터 로드
    df = generate_sales_data(n_months=60)

    # 2. 스케일링 (학습 안정화)
    scaler = StandardScaler()
    sales_scaled = scaler.fit_transform(df[['sales']].values).flatten()

    # 3. 시퀀스 생성 (스케일된 값으로 학습)
    X, y = create_sequences(sales_scaled, lookback=12)

    # 4. 학습/테스트 분할 (시간 순서 유지)
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    print(f"\n학습 데이터: {len(X_train)}개")
    print(f"테스트 데이터: {len(X_test)}개")

    # 5. 모델 학습
    model = train_diffusion(X_train, y_train, epochs=100)

    # 6. 마지막 시퀀스로 예측 (스케일된 값)
    last_sequence = X_test[-1]
    forecast_results_scaled = forecast_with_uncertainty(model, last_sequence, n_samples=100)

    # 7. 원 단위로 역변환
    samples_units = scaler.inverse_transform(forecast_results_scaled['samples'].reshape(-1, 1)).flatten()
    p5 = float(np.percentile(samples_units, 5))
    p25 = float(np.percentile(samples_units, 25))
    p50 = float(np.percentile(samples_units, 50))
    p75 = float(np.percentile(samples_units, 75))
    p95 = float(np.percentile(samples_units, 95))

    forecast_results = {
        'samples': samples_units,
        'p5': p5,
        'p25': p25,
        'p50': p50,
        'p75': p75,
        'p95': p95
    }

    # 8. 시각화 (원 단위)
    visualize_forecast(df, forecast_results, n_history=24)

    # 9. 리스크 분석
    risk_table = create_risk_analysis(forecast_results)

    print("\n" + "="*60)
    print("실습 완료!")
    print("="*60)
    print(f"출력 디렉토리: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
