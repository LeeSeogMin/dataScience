#!/usr/bin/env python3
"""
Autoencoder + XGBoost 하이브리드 실험
--------------------------------------------------
- 복잡한 합성 데이터(practice/chapter7/data/synthetic_complex.csv)를
  통해 오토인코더의 잠재 표현이 트리 모델 성능에 기여하는 정도를 테스트
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier, callback as xgb_callback


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def get_latent(self, x):
        return self.encoder(x)


def train_autoencoder(model: Autoencoder, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
                      epochs: int = 60, lr: float = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch, in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch, in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * batch.size(0)

        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"[AE] Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f}")

    return model


def extract_latent(model: Autoencoder, data: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.from_numpy(data).to(device)
    with torch.no_grad():
        latent = model.get_latent(tensor)
    return latent.cpu().numpy()


def evaluate_xgb(X_train, y_train, X_valid, y_valid, X_test, y_test):
    start = time.time()
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.7,
        random_state=42,
        eval_metric="auc",
        use_label_encoder=False,
        verbosity=0,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    elapsed = time.time() - start
    pred = xgb.predict(X_test)
    prob = xgb.predict_proba(X_test)[:, 1]
    return accuracy_score(y_test, pred), roc_auc_score(y_test, prob), elapsed


def load_synthetic_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run practice/chapter7/code/7-3b-synthetic-comparison.py to generate it.")
    df = pd.read_csv(path)
    X = df.drop(columns="target").values.astype(np.float32)
    y = df["target"].values.astype(np.int64)
    return X, y


def build_dataloader(X: np.ndarray, batch_size: int = 512):
    tensor = torch.from_numpy(X)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def main():
    set_seed(42)
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir.parent / "data"
    output_dir = data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "synthetic_complex.csv"

    X, y = load_synthetic_data(data_path)
    print(f"데이터 로드 완료: {data_path} | 샘플 수={len(X)}, 특성 수={X.shape[1]}")

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    print(f"학습/검증/테스트 분할: {len(X_train)}/{len(X_valid)}/{len(X_test)}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Autoencoder 학습 장치: {device}\n")

    train_loader = build_dataloader(X_train_scaled)
    valid_loader = build_dataloader(X_valid_scaled)

    autoencoder = Autoencoder(input_dim=X_train_scaled.shape[1], latent_dim=32)
    autoencoder = train_autoencoder(autoencoder, train_loader, valid_loader, device, epochs=60)

    latent_train = extract_latent(autoencoder, X_train_scaled, device)
    latent_valid = extract_latent(autoencoder, X_valid_scaled, device)
    latent_test = extract_latent(autoencoder, X_test_scaled, device)

    latent_scaler = StandardScaler()
    latent_train_scaled = latent_scaler.fit_transform(latent_train)
    latent_valid_scaled = latent_scaler.transform(latent_valid)
    latent_test_scaled = latent_scaler.transform(latent_test)

    combos = [
        ("원본 특성", X_train_scaled, X_valid_scaled, X_test_scaled),
        ("잠재 표현", latent_train_scaled, latent_valid_scaled, latent_test_scaled),
        ("원본 + 잠재", np.hstack([X_train_scaled, latent_train_scaled]),
         np.hstack([X_valid_scaled, latent_valid_scaled]),
         np.hstack([X_test_scaled, latent_test_scaled]))
    ]

    records = []
    for label, X_tr, X_val, X_te in combos:
        print(f"\n[label] {label}로 XGBoost 학습 중...")
        acc, auc, elapsed = evaluate_xgb(X_tr, y_train, X_val, y_valid, X_te, y_test)
        records.append({
            "feature_set": label,
            "feature_count": X_tr.shape[1],
            "accuracy": acc,
            "auc": auc,
            "time": elapsed
        })
        print(f"    정확도={acc:.4f}, AUC={auc:.4f}, 학습 시간={elapsed:.2f}초")

    results_path = output_dir / "ae_xgboost_hybrid_results.csv"
    pd.DataFrame(records).to_csv(results_path, index=False)
    print(f"\n결과 저장: {results_path}")


if __name__ == "__main__":
    main()
