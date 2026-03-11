"""
7장 실습: AE + XGBoost 하이브리드 - Semi-supervised 시나리오
- 레이블이 부족한 상황에서 잠재 표현의 가치 검증
- AE: 전체 데이터(50,000건)로 비지도 학습
- XGBoost: 소량의 레이블 데이터(1,000건)로 학습
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
# 1. Autoencoder 모델 (GPU 지원)
# ============================================================

class Autoencoder(nn.Module):
    """Deep Autoencoder for tabular data"""
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
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

        if valid_loss < best_loss:
            best_loss = valid_loss
            patience_counter = 0
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 15:
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
# 2. 실험 함수
# ============================================================

def run_experiment(X_all, y_all, X_test, y_test,
                   n_labeled, latent_dim, device, seed=42):
    """
    Semi-supervised 실험
    - X_all: 전체 학습 데이터 (AE 학습용)
    - n_labeled: XGBoost 학습에 사용할 레이블 데이터 수
    """
    np.random.seed(seed)

    # 레이블 데이터 샘플링 (층화 추출)
    if n_labeled < len(X_all):
        X_labeled, _, y_labeled, _ = train_test_split(
            X_all, y_all, train_size=n_labeled,
            random_state=seed, stratify=y_all
        )
    else:
        X_labeled, y_labeled = X_all, y_all

    # AE 학습용 데이터 분할 (전체 데이터 사용)
    X_ae_train, X_ae_valid = train_test_split(
        X_all, test_size=0.1, random_state=seed
    )

    # DataLoader
    train_dataset = TensorDataset(torch.FloatTensor(X_ae_train))
    valid_dataset = TensorDataset(torch.FloatTensor(X_ae_valid))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512)

    # Autoencoder 학습
    ae = Autoencoder(input_dim=X_all.shape[1], latent_dim=latent_dim)
    ae, recon_loss = train_autoencoder(
        ae, train_loader, valid_loader,
        epochs=100, lr=1e-3, device=device
    )

    # 잠재 표현 추출
    Z_labeled = extract_latent(ae, X_labeled, device)
    Z_test = extract_latent(ae, X_test, device)

    results = {}

    # 1) 원본 특성만 사용
    xgb_orig = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method='hist', device='cuda',
        random_state=seed, verbosity=0
    )
    xgb_orig.fit(X_labeled, y_labeled)
    pred_orig = xgb_orig.predict(X_test)
    prob_orig = xgb_orig.predict_proba(X_test)[:, 1]
    results['original'] = {
        'accuracy': accuracy_score(y_test, pred_orig),
        'auc': roc_auc_score(y_test, prob_orig)
    }

    # 2) 잠재 표현만 사용
    xgb_latent = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method='hist', device='cuda',
        random_state=seed, verbosity=0
    )
    xgb_latent.fit(Z_labeled, y_labeled)
    pred_latent = xgb_latent.predict(Z_test)
    prob_latent = xgb_latent.predict_proba(Z_test)[:, 1]
    results['latent'] = {
        'accuracy': accuracy_score(y_test, pred_latent),
        'auc': roc_auc_score(y_test, prob_latent)
    }

    # 3) 원본 + 잠재
    X_labeled_combined = np.hstack([X_labeled, Z_labeled])
    X_test_combined = np.hstack([X_test, Z_test])

    xgb_combined = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method='hist', device='cuda',
        random_state=seed, verbosity=0
    )
    xgb_combined.fit(X_labeled_combined, y_labeled)
    pred_comb = xgb_combined.predict(X_test_combined)
    prob_comb = xgb_combined.predict_proba(X_test_combined)[:, 1]
    results['combined'] = {
        'accuracy': accuracy_score(y_test, pred_comb),
        'auc': roc_auc_score(y_test, prob_comb)
    }

    return results, recon_loss


# ============================================================
# 3. 메인 실행
# ============================================================

def main():
    print("=" * 70)
    print("Semi-supervised 시나리오: 잠재 표현의 가치 검증")
    print("=" * 70)

    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 데이터 로드
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(data_dir, 'data', 'synthetic_complex.csv')

    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1).values.astype(np.float32)
    y = df['target'].values.astype(np.int64)

    print(f"전체 데이터: {X.shape[0]}건, {X.shape[1]}특성")

    # 테스트 데이터 분리
    X_train_all, X_test, y_train_all, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 스케일링
    scaler = StandardScaler()
    X_train_all = scaler.fit_transform(X_train_all)
    X_test = scaler.transform(X_test)

    print(f"학습 풀: {len(X_train_all)}, 테스트: {len(X_test)}")

    # 다양한 레이블 데이터 크기로 실험
    label_sizes = [500, 1000, 2000, 5000, 10000, 30000]
    latent_dim = 32

    print(f"\n잠재 차원: {latent_dim}")
    print(f"AE 학습 데이터: {len(X_train_all)}건 (비지도)")
    print("\n" + "-" * 70)

    all_results = []

    for n_labeled in label_sizes:
        print(f"\n레이블 데이터: {n_labeled}건으로 XGBoost 학습")

        results, recon_loss = run_experiment(
            X_train_all, y_train_all, X_test, y_test,
            n_labeled=n_labeled, latent_dim=latent_dim, device=device
        )

        print(f"  재구성 손실: {recon_loss:.6f}")
        print(f"  원본 특성    - 정확도: {results['original']['accuracy']:.4f}, AUC: {results['original']['auc']:.4f}")
        print(f"  잠재 표현    - 정확도: {results['latent']['accuracy']:.4f}, AUC: {results['latent']['auc']:.4f}")
        print(f"  원본 + 잠재  - 정확도: {results['combined']['accuracy']:.4f}, AUC: {results['combined']['auc']:.4f}")

        all_results.append({
            'n_labeled': n_labeled,
            'orig_acc': results['original']['accuracy'],
            'orig_auc': results['original']['auc'],
            'latent_acc': results['latent']['accuracy'],
            'latent_auc': results['latent']['auc'],
            'combined_acc': results['combined']['accuracy'],
            'combined_auc': results['combined']['auc']
        })

    # 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약: 레이블 데이터 크기에 따른 AUC 비교")
    print("=" * 70)
    print(f"{'레이블 수':>10} | {'원본':>8} | {'잠재':>8} | {'원본+잠재':>10} | {'잠재 이득':>10}")
    print("-" * 70)

    for r in all_results:
        gain = r['combined_auc'] - r['orig_auc']
        gain_str = f"+{gain:.4f}" if gain > 0 else f"{gain:.4f}"
        print(f"{r['n_labeled']:>10} | {r['orig_auc']:>8.4f} | {r['latent_auc']:>8.4f} | {r['combined_auc']:>10.4f} | {gain_str:>10}")

    # 결과 저장
    results_df = pd.DataFrame(all_results)
    output_dir = os.path.join(data_dir, 'data', 'output')
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(os.path.join(output_dir, 'semisupervised_results.csv'), index=False)
    print(f"\n결과 저장: {output_dir}/semisupervised_results.csv")

    return all_results


if __name__ == "__main__":
    results = main()
