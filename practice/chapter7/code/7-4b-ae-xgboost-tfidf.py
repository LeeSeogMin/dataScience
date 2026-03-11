"""
7장 실습: AE + XGBoost 하이브리드 - 고차원 텍스트 특성
- NSMC (Naver Sentiment Movie Corpus) 한국어 영화 리뷰
- TF-IDF로 고차원 희소 특성 생성 (5000+ 차원)
- AE로 저차원 잠재 표현 추출
- XGBoost 성능 비교: 원본 TF-IDF vs 잠재 vs 결합
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import urllib.request

# ============================================================
# 1. 데이터 다운로드 및 전처리
# ============================================================

def download_nsmc(data_dir):
    """NSMC 데이터 다운로드"""
    os.makedirs(data_dir, exist_ok=True)

    base_url = "https://raw.githubusercontent.com/e9t/nsmc/master/"
    files = {
        'train': 'ratings_train.txt',
        'test': 'ratings_test.txt'
    }

    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"다운로드 중: {filename}")
            urllib.request.urlretrieve(base_url + filename, filepath)

    return files


def load_nsmc(data_dir, sample_size=None):
    """NSMC 데이터 로드"""
    train_path = os.path.join(data_dir, 'ratings_train.txt')
    test_path = os.path.join(data_dir, 'ratings_test.txt')

    train_df = pd.read_csv(train_path, sep='\t')
    test_df = pd.read_csv(test_path, sep='\t')

    # 결측치 제거
    train_df = train_df.dropna(subset=['document'])
    test_df = test_df.dropna(subset=['document'])

    # 샘플링 (필요시)
    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(sample_size // 4, len(test_df)), random_state=42)

    return train_df, test_df


# ============================================================
# 2. Autoencoder 모델 (희소 입력 지원)
# ============================================================

class SparseAutoencoder(nn.Module):
    """Autoencoder for high-dimensional sparse features"""
    def __init__(self, input_dim, latent_dim=128):
        super().__init__()

        # Encoder - 점진적 차원 축소
        hidden1 = min(1024, input_dim // 2)
        hidden2 = min(512, hidden1 // 2)
        hidden3 = min(256, hidden2 // 2)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden3),
            nn.BatchNorm1d(hidden3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden3, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden2, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, input_dim)
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
        optimizer, mode='min', factor=0.5, patience=3
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            if patience_counter >= 10:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model, best_loss


def extract_latent(model, X, device, batch_size=512):
    """잠재 표현 추출"""
    model.eval()

    # scipy sparse matrix를 dense로 변환 (필요시)
    if hasattr(X, 'toarray'):
        X = X.toarray()

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
    print("고차원 텍스트 특성에서 잠재 표현의 가치 검증")
    print("데이터: NSMC (Naver Sentiment Movie Corpus)")
    print("=" * 70)

    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # 데이터 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(script_dir), 'data', 'nsmc')

    # NSMC 다운로드 및 로드
    print("\n1. 데이터 로드...")
    download_nsmc(data_dir)
    train_df, test_df = load_nsmc(data_dir, sample_size=50000)

    print(f"   학습 데이터: {len(train_df)}건")
    print(f"   테스트 데이터: {len(test_df)}건")

    # TF-IDF 특성 추출
    print("\n2. TF-IDF 특성 추출...")
    tfidf = TfidfVectorizer(
        max_features=5000,  # 고차원 특성
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf.fit_transform(train_df['document']).toarray().astype(np.float32)
    X_test_tfidf = tfidf.transform(test_df['document']).toarray().astype(np.float32)
    y_train = train_df['label'].values.astype(np.int64)
    y_test = test_df['label'].values.astype(np.int64)

    print(f"   TF-IDF 차원: {X_train_tfidf.shape[1]}")
    print(f"   희소성: {(X_train_tfidf == 0).mean():.1%}")

    # 학습/검증 분할
    X_train, X_valid, y_train_split, y_valid = train_test_split(
        X_train_tfidf, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    # Autoencoder 학습
    print("\n3. Autoencoder 학습...")
    latent_dim = 128
    print(f"   잠재 차원: {latent_dim}")

    train_dataset = TensorDataset(torch.FloatTensor(X_train))
    valid_dataset = TensorDataset(torch.FloatTensor(X_valid))
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256)

    ae = SparseAutoencoder(input_dim=X_train.shape[1], latent_dim=latent_dim)
    ae, recon_loss = train_autoencoder(
        ae, train_loader, valid_loader,
        epochs=100, lr=1e-3, device=device
    )
    print(f"   최종 재구성 손실: {recon_loss:.6f}")

    # 잠재 표현 추출
    print("\n4. 잠재 표현 추출...")
    Z_train = extract_latent(ae, X_train_tfidf, device)
    Z_test = extract_latent(ae, X_test_tfidf, device)
    print(f"   잠재 표현 shape: {Z_train.shape}")

    # XGBoost 실험
    print("\n5. XGBoost 분류 실험...")
    results = {}

    # 5-1) 원본 TF-IDF
    print("\n   [원본 TF-IDF 특성]")
    xgb_orig = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        tree_method='hist', device='cuda',
        random_state=42, verbosity=0
    )
    xgb_orig.fit(X_train_tfidf, y_train)
    pred_orig = xgb_orig.predict(X_test_tfidf)
    prob_orig = xgb_orig.predict_proba(X_test_tfidf)[:, 1]
    results['original'] = {
        'accuracy': accuracy_score(y_test, pred_orig),
        'auc': roc_auc_score(y_test, prob_orig),
        'n_features': X_train_tfidf.shape[1]
    }
    print(f"   정확도: {results['original']['accuracy']:.4f}")
    print(f"   AUC: {results['original']['auc']:.4f}")
    print(f"   특성 수: {results['original']['n_features']}")

    # 5-2) 잠재 표현만
    print("\n   [잠재 표현]")
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
        'n_features': Z_train.shape[1]
    }
    print(f"   정확도: {results['latent']['accuracy']:.4f}")
    print(f"   AUC: {results['latent']['auc']:.4f}")
    print(f"   특성 수: {results['latent']['n_features']}")

    # 5-3) 원본 + 잠재
    print("\n   [원본 + 잠재]")
    X_train_combined = np.hstack([X_train_tfidf, Z_train])
    X_test_combined = np.hstack([X_test_tfidf, Z_test])

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
    print(f"   특성 수: {results['combined']['n_features']}")

    # 결과 요약
    print("\n" + "=" * 70)
    print("결과 요약")
    print("=" * 70)
    print(f"{'특성 세트':<15} | {'특성 수':>8} | {'정확도':>8} | {'AUC':>8}")
    print("-" * 50)
    print(f"{'원본 TF-IDF':<15} | {results['original']['n_features']:>8} | {results['original']['accuracy']:>8.4f} | {results['original']['auc']:>8.4f}")
    print(f"{'잠재 표현':<15} | {results['latent']['n_features']:>8} | {results['latent']['accuracy']:>8.4f} | {results['latent']['auc']:>8.4f}")
    print(f"{'원본 + 잠재':<15} | {results['combined']['n_features']:>8} | {results['combined']['accuracy']:>8.4f} | {results['combined']['auc']:>8.4f}")

    # 차원 축소 효율성
    compression_ratio = results['original']['n_features'] / results['latent']['n_features']
    auc_retention = results['latent']['auc'] / results['original']['auc']
    print(f"\n압축률: {compression_ratio:.1f}x ({results['original']['n_features']} → {results['latent']['n_features']})")
    print(f"AUC 유지율: {auc_retention:.1%}")

    # 결과 저장
    output_dir = os.path.join(os.path.dirname(script_dir), 'data', 'output')
    os.makedirs(output_dir, exist_ok=True)

    results_df = pd.DataFrame([
        {'feature_set': 'original', **results['original']},
        {'feature_set': 'latent', **results['latent']},
        {'feature_set': 'combined', **results['combined']}
    ])
    results_df.to_csv(os.path.join(output_dir, 'tfidf_ae_results.csv'), index=False)
    print(f"\n결과 저장: {output_dir}/tfidf_ae_results.csv")

    return results


if __name__ == "__main__":
    results = main()
