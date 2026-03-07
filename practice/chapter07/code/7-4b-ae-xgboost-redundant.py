"""
7장 실습: AE + XGBoost 하이브리드 - 고차원 중복 특성
- 핵심 특성 100개 + 파생 특성 900개 (중복/상관)
- AE로 중복 제거하여 핵심 정보만 추출
- GPU 사용
"""

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

# ============================================================
# 1. 고차원 중복 데이터 생성
# ============================================================

def generate_redundant_data(n_samples=50000, n_core=100, n_derived=900,
                            noise_level=0.3, seed=42):
    """
    고차원 중복 특성 데이터 생성
    - n_core: 핵심 특성 수 (실제 정보 포함)
    - n_derived: 파생 특성 수 (핵심 특성의 선형 조합 + 노이즈)
    """
    np.random.seed(seed)

    # 1) 핵심 특성 생성
    X_core = np.random.randn(n_samples, n_core).astype(np.float32)

    # 2) 타겟 생성 (핵심 특성 기반)
    # 비선형 상호작용 포함
    y_prob = (
        0.3 * X_core[:, 0] * X_core[:, 1] +
        0.2 * np.sin(X_core[:, 2] * 2) +
        0.2 * X_core[:, 3] ** 2 +
        0.15 * X_core[:, 4:10].sum(axis=1) +
        0.1 * (X_core[:, 10] > 0).astype(float) * X_core[:, 11] +
        0.05 * X_core[:, 12:20].mean(axis=1)
    )
    y_prob = 1 / (1 + np.exp(-y_prob))  # sigmoid
    y = (np.random.rand(n_samples) < y_prob).astype(np.int64)

    # 3) 파생 특성 생성 (핵심 특성의 선형 조합 + 노이즈)
    # 각 파생 특성은 2~5개 핵심 특성의 조합
    X_derived = np.zeros((n_samples, n_derived), dtype=np.float32)

    for i in range(n_derived):
        # 랜덤하게 2~5개 핵심 특성 선택
        n_selected = np.random.randint(2, 6)
        selected_idx = np.random.choice(n_core, n_selected, replace=False)

        # 랜덤 가중치로 선형 조합
        weights = np.random.randn(n_selected)
        weights = weights / np.abs(weights).sum()  # 정규화

        X_derived[:, i] = X_core[:, selected_idx] @ weights

        # 노이즈 추가
        X_derived[:, i] += noise_level * np.random.randn(n_samples)

    # 4) 전체 특성 결합
    X = np.hstack([X_core, X_derived])

    return X, y, n_core


# ============================================================
# 2. Autoencoder 모델
# ============================================================

class RedundantAutoencoder(nn.Module):
    """Autoencoder for high-dimensional redundant features"""
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


def train_autoencoder(model, train_loader, valid_loader, epochs, lr, device):
    """Autoencoder 학습"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()

    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, in train_loader:
            X_batch = X_batch.to(device)
            optimizer.zero_grad()
            recon, _ = model(X_batch)
            loss = criterion(recon, X_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X_batch, in valid_loader:
                X_batch = X_batch.to(device)
                recon, _ = model(X_batch)
                loss = criterion(recon, X_batch)
                valid_loss += loss.item()

        valid_loss /= len(valid_loader)
        scheduler.step(valid_loss)

        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: train_loss={train_loss:.6f}, valid_loss={valid_loss:.6f}")

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 15:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model, best_loss


def extract_latent(model, X, device, batch_size=1024):
    """잠재 표현 추출"""
    model.eval()
    dataset = TensorDataset(torch.FloatTensor(X))
    loader = DataLoader(dataset, batch_size=batch_size)

    latents = []
    with torch.no_grad():
        for X_batch, in loader:
            X_batch = X_batch.to(device)
            _, z = model(X_batch)
            latents.append(z.cpu().numpy())

    return np.vstack(latents)


# ============================================================
# 3. 메인 실험
# ============================================================

def main():
    print("=" * 70)
    print("고차원 중복 특성에서 잠재 표현의 가치 검증")
    print("=" * 70)

    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 데이터 생성
    print("\n1. 데이터 생성...")
    n_core = 100
    n_derived = 900
    X, y, _ = generate_redundant_data(
        n_samples=50000,
        n_core=n_core,
        n_derived=n_derived,
        noise_level=0.3
    )
    print(f"   전체 특성: {X.shape[1]} (핵심: {n_core}, 파생: {n_derived})")
    print(f"   샘플 수: {X.shape[0]}")
    print(f"   클래스 비율: {y.mean():.2%} positive")

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 스케일링
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # 학습/검증 분할
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )

    # Autoencoder 학습
    print("\n2. Autoencoder 학습...")
    latent_dim = 64  # 핵심 특성 수보다 작게
    print(f"   잠재 차원: {latent_dim}")

    train_dataset = TensorDataset(torch.FloatTensor(X_tr))
    valid_dataset = TensorDataset(torch.FloatTensor(X_val))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256)

    ae = RedundantAutoencoder(input_dim=X_train.shape[1], latent_dim=latent_dim)
    ae, recon_loss = train_autoencoder(
        ae, train_loader, valid_loader,
        epochs=100, lr=1e-3, device=device
    )
    print(f"   최종 재구성 손실: {recon_loss:.6f}")

    # 잠재 표현 추출
    print("\n3. 잠재 표현 추출...")
    Z_train = extract_latent(ae, X_train, device)
    Z_test = extract_latent(ae, X_test, device)

    # 핵심 특성만 추출 (비교용)
    X_train_core = X_train[:, :n_core]
    X_test_core = X_test[:, :n_core]

    # XGBoost 실험
    print("\n4. XGBoost 분류 실험...")
    results = {}

    # 4-1) 전체 특성 (1000개)
    print("\n   [전체 특성 - 1000개]")
    xgb_all = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method='hist', device='cuda',
        random_state=42, verbosity=0
    )
    xgb_all.fit(X_train, y_train)
    pred_all = xgb_all.predict(X_test)
    prob_all = xgb_all.predict_proba(X_test)[:, 1]
    results['all'] = {
        'accuracy': accuracy_score(y_test, pred_all),
        'auc': roc_auc_score(y_test, prob_all),
        'n_features': X_train.shape[1]
    }
    print(f"   정확도: {results['all']['accuracy']:.4f}")
    print(f"   AUC: {results['all']['auc']:.4f}")

    # 4-2) 핵심 특성만 (100개) - Oracle
    print("\n   [핵심 특성만 - 100개 (Oracle)]")
    xgb_core = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method='hist', device='cuda',
        random_state=42, verbosity=0
    )
    xgb_core.fit(X_train_core, y_train)
    pred_core = xgb_core.predict(X_test_core)
    prob_core = xgb_core.predict_proba(X_test_core)[:, 1]
    results['core'] = {
        'accuracy': accuracy_score(y_test, pred_core),
        'auc': roc_auc_score(y_test, prob_core),
        'n_features': n_core
    }
    print(f"   정확도: {results['core']['accuracy']:.4f}")
    print(f"   AUC: {results['core']['auc']:.4f}")

    # 4-3) 잠재 표현 (64개)
    print("\n   [잠재 표현 - 64개]")
    xgb_latent = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method='hist', device='cuda',
        random_state=42, verbosity=0
    )
    xgb_latent.fit(Z_train, y_train)
    pred_latent = xgb_latent.predict(Z_test)
    prob_latent = xgb_latent.predict_proba(Z_test)[:, 1]
    results['latent'] = {
        'accuracy': accuracy_score(y_test, pred_latent),
        'auc': roc_auc_score(y_test, prob_latent),
        'n_features': latent_dim
    }
    print(f"   정확도: {results['latent']['accuracy']:.4f}")
    print(f"   AUC: {results['latent']['auc']:.4f}")

    # 4-4) 전체 + 잠재
    print("\n   [전체 + 잠재 - 1064개]")
    X_train_combined = np.hstack([X_train, Z_train])
    X_test_combined = np.hstack([X_test, Z_test])

    xgb_combined = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method='hist', device='cuda',
        random_state=42, verbosity=0
    )
    xgb_combined.fit(X_train_combined, y_train)
    pred_comb = xgb_combined.predict(X_test_combined)
    prob_comb = xgb_combined.predict_proba(X_test_combined)[:, 1]
    results['combined'] = {
        'accuracy': accuracy_score(y_test, pred_comb),
        'auc': roc_auc_score(y_test, prob_comb),
        'n_features': X_train_combined.shape[1]
    }
    print(f"   정확도: {results['combined']['accuracy']:.4f}")
    print(f"   AUC: {results['combined']['auc']:.4f}")

    # 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"{'특성 세트':<20} | {'특성 수':>8} | {'정확도':>8} | {'AUC':>8}")
    print("-" * 55)
    print(f"{'전체 (1000개)':<20} | {results['all']['n_features']:>8} | {results['all']['accuracy']:>8.4f} | {results['all']['auc']:>8.4f}")
    print(f"{'핵심만 (Oracle)':<20} | {results['core']['n_features']:>8} | {results['core']['accuracy']:>8.4f} | {results['core']['auc']:>8.4f}")
    print(f"{'잠재 표현 (64개)':<20} | {results['latent']['n_features']:>8} | {results['latent']['accuracy']:>8.4f} | {results['latent']['auc']:>8.4f}")
    print(f"{'전체 + 잠재':<20} | {results['combined']['n_features']:>8} | {results['combined']['accuracy']:>8.4f} | {results['combined']['auc']:>8.4f}")

    # 효율성 분석
    print("\n효율성 분석:")
    print(f"  압축률: {results['all']['n_features'] / results['latent']['n_features']:.1f}x (1000 → 64)")
    print(f"  잠재 vs 전체: AUC {results['latent']['auc']:.4f} vs {results['all']['auc']:.4f}")

    if results['latent']['auc'] >= results['all']['auc'] * 0.95:
        print(f"  → 잠재 표현이 전체 특성 대비 95% 이상 성능 유지!")

    if results['combined']['auc'] > results['all']['auc']:
        improvement = results['combined']['auc'] - results['all']['auc']
        print(f"  → 잠재 표현 추가로 AUC +{improvement:.4f} 향상!")

    # 결과 저장
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(os.path.dirname(script_dir), 'data', 'output')
    os.makedirs(output_dir, exist_ok=True)

    results_df = pd.DataFrame([
        {'feature_set': 'all', **results['all']},
        {'feature_set': 'core', **results['core']},
        {'feature_set': 'latent', **results['latent']},
        {'feature_set': 'combined', **results['combined']}
    ])
    results_df.to_csv(os.path.join(output_dir, 'redundant_ae_results.csv'), index=False)
    print(f"\n결과 저장: {output_dir}/redundant_ae_results.csv")

    return results


if __name__ == "__main__":
    results = main()
