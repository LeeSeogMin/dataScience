"""
8.3 VAE 데이터 증강과 이상치 탐지 실습

목표:
- VAE로 소수 클래스 데이터 증강
- 재구성 오차 기반 이상치 탐지
- 잠재공간 시각화

데이터: 신용카드 사기 거래 (불균형 데이터)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

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
OUTPUT_DIR = DATA_DIR / "output" / "8-3-vae-practice"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VAE 모델
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim=2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_logvar = nn.Linear(8, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE 손실 함수"""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld

def load_credit_fraud_data():
    """신용카드 사기 데이터 로드"""
    print("데이터 로드 중...")
    df = pd.read_csv(INPUT_DIR / "credit_fraud.csv")
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["fraud"].values.astype(int)

    counts = pd.Series(y).value_counts().to_dict()
    print(f"데이터 로드 완료: {len(df)} 샘플")
    print(f"클래스 분포: {counts}")
    return X, y, feature_cols

def train_vae(X_fraud, epochs=50):
    """VAE 학습"""
    print("\n" + "="*60)
    print("VAE 학습")
    print("="*60)

    device = torch.device('cpu')
    input_dim = X_fraud.shape[1]

    # 데이터 준비
    X_tensor = torch.FloatTensor(X_fraud)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 모델 초기화
    vae = VAE(input_dim, latent_dim=2).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

    # 학습
    print(f"학습 시작 (Epochs={epochs})...")
    for epoch in range(epochs):
        vae.train()
        train_loss = 0
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)

            optimizer.zero_grad()
            recon, mu, logvar = vae(batch_x)
            loss = vae_loss(recon, batch_x, mu, logvar)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            avg_loss = train_loss / len(loader.dataset)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    print("학습 완료")
    return vae

def augment_fraud_data(vae, X_fraud, n_samples):
    """VAE로 합성 샘플 생성 - 학습된 사기 패턴 주변에서 샘플링"""
    print(f"\n합성 샘플 생성: {n_samples}개")

    vae.eval()
    with torch.no_grad():
        # 1. 원본 사기 샘플을 잠재공간으로 인코딩
        X_tensor = torch.FloatTensor(X_fraud)
        mu, logvar = vae.encode(X_tensor)

        # 2. 사기 샘플들의 잠재 분포 파악 (평균과 표준편차)
        z_mean = mu.mean(dim=0)
        z_std = mu.std(dim=0) + 0.1  # 약간의 여유 추가

        # 3. 해당 분포에서 새로운 잠재벡터 샘플링
        z = z_mean + z_std * torch.randn(n_samples, vae.latent_dim)

        # 4. 디코딩하여 합성 샘플 생성
        synthetic = vae.decode(z).numpy()

    print("생성 완료")
    return synthetic

def evaluate_augmentation(X, y):
    """데이터 증강 효과 평가"""
    print("\n" + "="*60)
    print("1. 데이터 증강 효과 평가")
    print("="*60)

    # Train-test split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 불균형 시나리오: 사기 샘플을 30개만 사용
    # 실무에서 "희귀 사기 유형"이나 "신규 사기 패턴"을 시뮬레이션
    X_normal = X_train_full[y_train_full == 0]
    X_fraud_full = X_train_full[y_train_full == 1]

    # 사기 샘플 30개만 선택 (VAE 학습 가능한 최소 수준)
    n_fraud_limited = 30
    np.random.seed(42)
    fraud_indices = np.random.choice(len(X_fraud_full), n_fraud_limited, replace=False)
    X_fraud_limited = X_fraud_full[fraud_indices]

    X_train = np.vstack([X_normal, X_fraud_limited])
    y_train = np.hstack([np.zeros(len(X_normal)), np.ones(n_fraud_limited)])

    print(f"\n[불균형 시나리오]")
    print(f"학습 데이터: 정상 {len(X_normal)}개, 사기 {n_fraud_limited}개 (비율 {len(X_normal)/n_fraud_limited:.0f}:1)")

    print("\n[Baseline] 원본 데이터만 사용:")
    clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_baseline.fit(X_train, y_train)
    y_pred_baseline = clf_baseline.predict(X_test)

    acc_baseline = accuracy_score(y_test, y_pred_baseline)
    f1_baseline = f1_score(y_test, y_pred_baseline)
    recall_baseline = recall_score(y_test, y_pred_baseline)

    print(f"Accuracy: {acc_baseline:.4f}")
    print(f"F1-Score: {f1_baseline:.4f}")
    print(f"Recall (사기 탐지율): {recall_baseline:.4f}")

    # 사기 거래만으로 VAE 학습 → 합성 사기 샘플 생성
    X_fraud = X_fraud_limited  # 10개의 사기 샘플로 VAE 학습
    vae = train_vae(X_fraud, epochs=100)  # 에폭 증가
    synthetic_fraud = augment_fraud_data(vae, X_fraud, n_samples=200)  # 200개 생성

    # 증강된 데이터로 학습
    X_augmented = np.vstack([X_train, synthetic_fraud])
    y_augmented = np.hstack([y_train, np.ones(len(synthetic_fraud))])

    print("\n[Augmented] 원본 + 합성 데이터 사용:")
    clf_augmented = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_augmented.fit(X_augmented, y_augmented)
    y_pred_augmented = clf_augmented.predict(X_test)

    acc_augmented = accuracy_score(y_test, y_pred_augmented)
    f1_augmented = f1_score(y_test, y_pred_augmented)
    recall_augmented = recall_score(y_test, y_pred_augmented)

    print(f"Accuracy: {acc_augmented:.4f}")
    print(f"F1-Score: {f1_augmented:.4f}")
    print(f"Recall (사기 탐지율): {recall_augmented:.4f}")

    # 개선도 계산
    print("\n[개선도]:")
    if f1_baseline > 0:
        f1_change = (f1_augmented/f1_baseline-1)*100
        print(f"F1-Score: {f1_baseline:.4f} → {f1_augmented:.4f} ({f1_change:+.1f}%)")
    else:
        print(f"F1-Score: {f1_baseline:.4f} → {f1_augmented:.4f} (Baseline=0, 절대 개선)")

    if recall_baseline > 0:
        recall_change = (recall_augmented/recall_baseline-1)*100
        print(f"Recall: {recall_baseline:.4f} → {recall_augmented:.4f} ({recall_change:+.1f}%)")
    else:
        print(f"Recall: {recall_baseline:.4f} → {recall_augmented:.4f} (Baseline=0, 절대 개선)")

    if f1_augmented > f1_baseline:
        print("✅ 데이터 증강 효과 확인 (F1 개선)")
    elif f1_augmented == f1_baseline and f1_baseline == 0:
        print("⚠️ 두 방법 모두 사기 탐지 실패 - 데이터/모델 튜닝 필요")
    else:
        print("⚠️ 추가 튜닝 필요")

    return {
        'baseline': (acc_baseline, f1_baseline, recall_baseline),
        'augmented': (acc_augmented, f1_augmented, recall_augmented),
        'vae_for_augmentation': vae,
        'X': X,
        'y': y
    }

def detect_anomalies(vae, X, y):
    """재구성 오차 기반 이상치 탐지"""
    print("\n" + "="*60)
    print("2. 이상치 탐지 (재구성 오차)")
    print("="*60)

    vae.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        recon, _, _ = vae(X_tensor)
        recon_errors = ((X_tensor - recon) ** 2).mean(dim=1).numpy()

    # 임계값: 정상(0) 재구성 오차의 95% 백분위수
    threshold = np.percentile(recon_errors[y == 0], 95)
    predictions = (recon_errors > threshold).astype(int)

    # 평가
    from sklearn.metrics import classification_report
    print("\n[이상치 탐지 성능]:")
    print(classification_report(y, predictions, target_names=['정상', '사기']))

    # 오탐률/미탐률 분석
    tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
    fpr = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate

    print(f"\n임계값: {threshold:.4f}")
    print(f"오탐률 (FPR): {fpr:.2%} - 정상 거래를 사기로 오인")
    print(f"미탐률 (FNR): {fnr:.2%} - 사기 거래를 정상으로 오인")

    if fpr < 0.1 and fnr < 0.3:
        print("✅ 이상치 탐지 성능 우수")
    else:
        print("⚠️ 임계값 조정 필요")

    return recon_errors, threshold

def visualize_latent_space(vae, X, y):
    """잠재공간 시각화 (흑백)"""
    print("\n잠재공간 시각화 생성 중...")

    vae.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        mu, _ = vae.encode(X_tensor)
        z = mu.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 잠재공간 산점도 (흑백: 마커 모양으로 구분)
    ax1 = axes[0]
    markers = ['o', 'x']  # 원형 vs X표시
    grays = ['0.7', '0.0']  # 회색 vs 검정
    labels = ['Normal', 'Fraud']
    for i, (marker, gray, label) in enumerate(zip(markers, grays, labels)):
        mask = y == i
        ax1.scatter(z[mask, 0], z[mask, 1], c=gray, marker=marker, label=label, alpha=0.7, s=30)

    ax1.set_xlabel('Latent Dimension 1')
    ax1.set_ylabel('Latent Dimension 2')
    ax1.set_title('VAE Latent Space (Normal vs Fraud)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 재구성 오차 분포 (흑백: 해칭 패턴으로 구분)
    ax2 = axes[1]
    vae.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        recon, _, _ = vae(X_tensor)
        recon_errors = ((X_tensor - recon) ** 2).mean(dim=1).numpy()

    hatches = ['', '///']  # 빈 패턴 vs 사선
    grays = ['0.7', '0.3']  # 밝은 회색 vs 어두운 회색
    for i, (hatch, gray, label) in enumerate(zip(hatches, grays, labels)):
        mask = y == i
        ax2.hist(recon_errors[mask], bins=30, alpha=0.7, label=label,
                 color=gray, edgecolor='black', hatch=hatch, density=True)

    threshold = np.percentile(recon_errors, 95)
    ax2.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold (95%ile)')

    ax2.set_xlabel('Reconstruction Error')
    ax2.set_ylabel('Density')
    ax2.set_title('Reconstruction Error Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = OUTPUT_DIR / "vae_latent_space_and_errors.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"저장 완료: {save_path}")
    plt.close()

def create_performance_comparison(results):
    """성능 비교 표 생성"""
    print("\n" + "="*60)
    print("종합 평가")
    print("="*60)

    baseline = results['baseline']
    augmented = results['augmented']

    # 개선도 계산 (division by zero 방지)
    if baseline[2] > 0:
        improvement = f"{(augmented[2]/baseline[2]-1)*100:+.1f}%"
    elif augmented[2] > 0:
        improvement = "∞ (0→양수)"
    else:
        improvement = "N/A"

    comparison = pd.DataFrame({
        '방법': ['원본 데이터만', '원본 + VAE 증강'],
        'Accuracy': [f"{baseline[0]:.4f}", f"{augmented[0]:.4f}"],
        'F1-Score': [f"{baseline[1]:.4f}", f"{augmented[1]:.4f}"],
        'Recall (사기탐지율)': [f"{baseline[2]:.4f}", f"{augmented[2]:.4f}"],
        '개선도': ['-', improvement]
    })

    print("\n" + comparison.to_string(index=False))

    comparison.to_csv(OUTPUT_DIR / "vae_performance_comparison.csv", index=False, encoding='utf-8-sig')
    print(f"\n성능 비교 저장: {OUTPUT_DIR / 'vae_performance_comparison.csv'}")

    return comparison

def main():
    print("="*60)
    print("8.3 VAE 데이터 증강과 이상치 탐지 실습")
    print("="*60)

    set_seed(42)
    configure_matplotlib(OUTPUT_DIR)

    # 1. 데이터 로드
    X, y, feature_cols = load_credit_fraud_data()

    # 2. 표준화
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. 데이터 증강 효과 평가
    results = evaluate_augmentation(X_scaled, y)

    # 4. 이상치 탐지 (정상 데이터만 학습)
    X_normal = X_scaled[y == 0]
    vae_normal = train_vae(X_normal, epochs=50)
    recon_errors, threshold = detect_anomalies(vae_normal, X_scaled, y)

    # 5. 시각화
    visualize_latent_space(vae_normal, X_scaled, y)

    # 6. 성능 비교
    comparison = create_performance_comparison(results)

    print("\n" + "="*60)
    print("실습 완료!")
    print("="*60)
    print(f"출력 디렉토리: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
