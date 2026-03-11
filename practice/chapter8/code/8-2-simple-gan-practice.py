"""
8.2 간단한 GAN으로 합성 데이터 생성 실습

목표:
- 간단한 GAN으로 정형 데이터 합성
- 통계적 유사도, ML 효용성 평가

데이터: Adult Income Dataset
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy import stats

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
OUTPUT_DIR = DATA_DIR / "output" / "8-2-gan-practice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 간단한 GAN 모델
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def load_data():
    """데이터 로드"""
    print("데이터 로드 중...")
    df = pd.read_csv(INPUT_DIR / "adult_income.csv")

    print(f"데이터 로드 완료: {len(df)} 샘플")
    print(f"Target 분포: {df['income'].value_counts().to_dict()}")

    return df

def train_gan(X_train, epochs=200, latent_dim=10):
    """GAN 학습"""
    print("\n" + "="*60)
    print("GAN 학습")
    print("="*60)

    device = torch.device('cpu')
    input_dim = X_train.shape[1]

    # 모델 초기화
    generator = Generator(latent_dim, input_dim).to(device)
    discriminator = Discriminator(input_dim).to(device)

    # 옵티마이저
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0002)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002)

    # 손실 함수
    criterion = nn.BCELoss()

    # 데이터 준비
    X_tensor = torch.FloatTensor(X_train)
    dataset = TensorDataset(X_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"학습 시작 (Epochs={epochs})...")
    for epoch in range(epochs):
        for real_data, in loader:
            batch_size = real_data.size(0)
            real_data = real_data.to(device)

            # ===== Train Discriminator =====
            d_optimizer.zero_grad()

            # Real data
            real_labels = torch.ones(batch_size, 1)
            d_real = discriminator(real_data)
            d_real_loss = criterion(d_real, real_labels)

            # Fake data
            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            fake_labels = torch.zeros(batch_size, 1)
            d_fake = discriminator(fake_data.detach())
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optimizer.step()

            # ===== Train Generator =====
            g_optimizer.zero_grad()

            z = torch.randn(batch_size, latent_dim)
            fake_data = generator(z)
            g_fake = discriminator(fake_data)
            g_loss = criterion(g_fake, real_labels)  # Generator wants D to output 1

            g_loss.backward()
            g_optimizer.step()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    print("학습 완료")
    return generator, latent_dim

def generate_synthetic(generator, latent_dim, n_samples):
    """합성 데이터 생성"""
    print(f"\n합성 데이터 생성: {n_samples} 샘플")

    generator.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        synthetic = generator(z).numpy()

    print(f"생성 완료: {len(synthetic)} 샘플")
    return synthetic

def evaluate_statistical_similarity(real, synthetic, continuous_cols):
    """통계적 유사도 평가"""
    print("\n" + "="*60)
    print("1. 통계적 유사도 평가")
    print("="*60)

    results = []

    for col_idx, col_name in continuous_cols:
        real_mean = real[:, col_idx].mean()
        syn_mean = synthetic[:, col_idx].mean()
        real_std = real[:, col_idx].std()
        syn_std = synthetic[:, col_idx].std()

        # KS 검정
        ks_stat, ks_pval = stats.ks_2samp(real[:, col_idx], synthetic[:, col_idx])

        results.append({
            'Variable': col_name,
            'Real Mean': f"{real_mean:.2f}",
            'Synthetic Mean': f"{syn_mean:.2f}",
            'Real Std': f"{real_std:.2f}",
            'Synthetic Std': f"{syn_std:.2f}",
            'KS p-value': f"{ks_pval:.4f}"
        })

    df_results = pd.DataFrame(results)
    print("\n연속형 변수 통계:")
    print(df_results.to_string(index=False))

    return df_results

def evaluate_ml_utility(real, synthetic, y_real, y_syn):
    """ML 효용성 평가 (TSTR)"""
    print("\n" + "="*60)
    print("2. ML 효용성 평가 (TSTR)")
    print("="*60)

    # Train on Real, Test on Real (TRTR)
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        real, y_real, test_size=0.3, random_state=42
    )

    clf_trtr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_trtr.fit(X_train_r, y_train_r)
    y_pred_trtr = clf_trtr.predict(X_test_r)
    acc_trtr = accuracy_score(y_test_r, y_pred_trtr)
    f1_trtr = f1_score(y_test_r, y_pred_trtr)

    # Train on Synthetic, Test on Real (TSTR)
    clf_tstr = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_tstr.fit(synthetic, y_syn)
    y_pred_tstr = clf_tstr.predict(X_test_r)
    acc_tstr = accuracy_score(y_test_r, y_pred_tstr)
    f1_tstr = f1_score(y_test_r, y_pred_tstr)

    tstr_ratio = acc_tstr / acc_trtr

    print(f"\nTRTR (Baseline): Accuracy={acc_trtr:.4f}, F1={f1_trtr:.4f}")
    print(f"TSTR (Synthetic): Accuracy={acc_tstr:.4f}, F1={f1_tstr:.4f}")
    print(f"TSTR Ratio: {tstr_ratio:.2%} (권장 ≥ 90%)")

    if tstr_ratio >= 0.9:
        print("✅ ML 효용성 우수")
    else:
        print("⚠️ ML 효용성 개선 필요")

    return {
        'TRTR_Accuracy': acc_trtr,
        'TSTR_Accuracy': acc_tstr,
        'TSTR_Ratio': tstr_ratio
    }

def visualize_distributions(real, synthetic, feature_names):
    """분포 시각화"""
    print("\n분포 비교 그래프 생성 중...")

    n_features = real.shape[1]
    n_cols = 2 if n_features <= 4 else 3
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle('Real vs Synthetic Feature Distributions', fontsize=16, fontweight='bold')
    axes = np.array(axes).reshape(-1)

    for idx in range(n_features):
        ax = axes[idx]
        ax.hist(real[:, idx], bins=30, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(synthetic[:, idx], bins=30, alpha=0.5, label='Synthetic', density=True, color='red')
        ax.set_xlabel(feature_names[idx])
        ax.set_ylabel('Density')
        ax.set_title(f'{feature_names[idx]} Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

    for ax in axes[n_features:]:
        ax.axis("off")

    plt.tight_layout()
    save_path = OUTPUT_DIR / "gan_distribution_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"저장 완료: {save_path}")
    plt.close()

def create_summary_table(stat_results, ml_results):
    """종합 평가 표 생성"""
    print("\n" + "="*60)
    print("종합 평가")
    print("="*60)

    ks_pvals = pd.to_numeric(stat_results["KS p-value"], errors="coerce")
    ks_pass_ratio = float((ks_pvals >= 0.05).mean()) if len(ks_pvals) else 0.0

    summary = pd.DataFrame(
        {
            '평가 차원': ['통계적 유사도', 'ML 효용성'],
            '지표': ['KS p-value', 'TSTR Ratio'],
            '결과': [
                f"p≥0.05 비율: {ks_pass_ratio:.1%}",
                f"{ml_results['TSTR_Ratio']:.1%}"
            ],
            '기준': ['p≥0.05', '≥ 90%'],
            '평가': [
                '✅ 통과' if ks_pass_ratio >= 0.8 else '⚠️ 개선 필요',
                '✅ 통과' if ml_results['TSTR_Ratio'] >= 0.9 else '⚠️ 개선 필요'
            ]
        }
    )

    print("\n" + summary.to_string(index=False))

    summary.to_csv(OUTPUT_DIR / "gan_evaluation_summary.csv", index=False, encoding='utf-8-sig')
    print(f"\n평가 결과 저장: {OUTPUT_DIR / 'gan_evaluation_summary.csv'}")

    return summary

def main():
    print("="*60)
    print("8.2 GAN 합성 데이터 생성 실습")
    print("="*60)

    set_seed(42)
    configure_matplotlib(OUTPUT_DIR)

    # 1. 데이터 로드
    df = load_data()

    # 2. 전처리
    continuous_cols = ['age', 'education_num', 'hours_per_week', 'capital_gain']
    target_col = "income"

    X = df[continuous_cols].values.astype(float)
    y = df[target_col].values.astype(int)

    train_cols = continuous_cols + [target_col]
    train_data = df[train_cols].values.astype(float)

    # 표준화
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_data)

    # 3. GAN 학습
    generator, latent_dim = train_gan(train_scaled, epochs=200)

    # 4. 합성 데이터 생성
    synthetic_scaled = generate_synthetic(generator, latent_dim, n_samples=len(train_scaled))
    synthetic_data = scaler.inverse_transform(synthetic_scaled)

    # 5. 합성 데이터에서 타겟 복원 (0/1로 이산화)
    X_syn = synthetic_data[:, :len(continuous_cols)]
    y_syn_cont = synthetic_data[:, -1]
    y_syn = (y_syn_cont >= 0.5).astype(int)

    # 6. 평가
    stat_results = evaluate_statistical_similarity(
        X,
        X_syn,
        [(i, name) for i, name in enumerate(continuous_cols)],
    )
    ml_results = evaluate_ml_utility(X, X_syn, y, y_syn)

    # 7. 시각화
    visualize_distributions(X, X_syn, continuous_cols)

    # 8. 종합 평가
    summary = create_summary_table(stat_results, ml_results)

    print("\n" + "="*60)
    print("실습 완료!")
    print("="*60)
    print(f"출력 디렉토리: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
