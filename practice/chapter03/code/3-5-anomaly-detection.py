"""
3-5-anomaly-detection.py: 이상치 탐지 기법 비교

4가지 알고리즘을 비교한다:
1. Isolation Forest
2. One-class SVM
3. Local Outlier Factor (LOF)
4. Autoencoder (PyTorch)

실행 방법:
    python 3-5-anomaly-detection.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    """이상치 탐지용 Autoencoder."""

    def __init__(self, input_dim: int, hidden_dim: int = 8, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(X_train: np.ndarray, epochs: int = 50, lr: float = 0.01) -> Autoencoder:
    """Autoencoder를 학습한다."""
    input_dim = X_train.shape[1]
    model = Autoencoder(input_dim)

    # 데이터 준비
    X_tensor = torch.FloatTensor(X_train)
    dataset = TensorDataset(X_tensor, X_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    # 학습
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_x, _ in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_x)
            loss.backward()
            optimizer.step()

    return model


def predict_autoencoder(model: Autoencoder, X: np.ndarray, contamination: float) -> np.ndarray:
    """Autoencoder로 이상치를 탐지한다."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X)
        X_reconstructed = model(X_tensor).numpy()

    # 재구성 오차 계산
    mse = np.mean((X - X_reconstructed) ** 2, axis=1)

    # 임계값: 상위 contamination% 비율을 이상치로 판단
    threshold = np.percentile(mse, (1 - contamination) * 100)
    y_pred = np.where(mse > threshold, -1, 1)

    return y_pred


def evaluate_detector(y_true, y_pred, method_name: str) -> dict:
    """이상치 탐지 성능을 평가한다."""
    # 이상치를 양성(1)으로 변환하여 평가
    y_true_binary = (y_true == -1).astype(int)
    y_pred_binary = (y_pred == -1).astype(int)

    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    cm = confusion_matrix(y_true_binary, y_pred_binary)

    return {
        'method': method_name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def demo_credit_card_fraud():
    """신용카드 사기 탐지 시뮬레이션: 4가지 알고리즘 비교."""
    print("\n" + "="*70)
    print("신용카드 사기 탐지: 4가지 알고리즘 비교")
    print("="*70)

    # 데이터 로드
    data_path = Path(__file__).parent.parent / "data" / "credit_card_fraud.csv"
    df = pd.read_csv(data_path)

    feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    X = df[feature_cols].values
    y = df['label'].values

    n_normal = (y == 1).sum()
    n_fraud = (y == -1).sum()
    contamination = n_fraud / len(y)

    print(f"데이터: {len(X):,}개 거래")
    print(f"사기: {n_fraud}건 ({contamination*100:.2f}%)")
    print(f"특성: {X.shape[1]}개")

    # 스케일링 (One-class SVM, Autoencoder용)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    # 랜덤 시드 설정
    np.random.seed(42)
    torch.manual_seed(42)

    # 1. Isolation Forest
    print("\n[1/4] Isolation Forest...")
    start_time = time.time()
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    y_pred_iso = iso_forest.fit_predict(X)
    time_iso = time.time() - start_time

    result = evaluate_detector(y, y_pred_iso, "Isolation Forest")
    result['time'] = time_iso
    results.append(result)
    print(f"   정밀도={result['precision']:.3f}, 재현율={result['recall']:.3f}, "
          f"F1={result['f1']:.3f}, 시간={time_iso:.3f}초")

    # 2. One-class SVM
    print("\n[2/4] One-class SVM...")
    start_time = time.time()
    ocsvm = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
    y_pred_svm = ocsvm.fit_predict(X_scaled)
    time_svm = time.time() - start_time

    result = evaluate_detector(y, y_pred_svm, "One-class SVM")
    result['time'] = time_svm
    results.append(result)
    print(f"   정밀도={result['precision']:.3f}, 재현율={result['recall']:.3f}, "
          f"F1={result['f1']:.3f}, 시간={time_svm:.3f}초")

    # 3. Local Outlier Factor
    print("\n[3/4] Local Outlier Factor...")
    start_time = time.time()
    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
    y_pred_lof = lof.fit_predict(X)
    time_lof = time.time() - start_time

    result = evaluate_detector(y, y_pred_lof, "Local Outlier Factor")
    result['time'] = time_lof
    results.append(result)
    print(f"   정밀도={result['precision']:.3f}, 재현율={result['recall']:.3f}, "
          f"F1={result['f1']:.3f}, 시간={time_lof:.3f}초")

    # 4. Autoencoder
    print("\n[4/4] Autoencoder (PyTorch)...")
    start_time = time.time()
    # 정상 데이터만으로 학습 (One-class 접근)
    X_normal_scaled = X_scaled[y == 1]
    ae_model = train_autoencoder(X_normal_scaled, epochs=50, lr=0.01)
    y_pred_ae = predict_autoencoder(ae_model, X_scaled, contamination)
    time_ae = time.time() - start_time

    result = evaluate_detector(y, y_pred_ae, "Autoencoder")
    result['time'] = time_ae
    results.append(result)
    print(f"   정밀도={result['precision']:.3f}, 재현율={result['recall']:.3f}, "
          f"F1={result['f1']:.3f}, 시간={time_ae:.3f}초")

    # 결과 요약 표
    print("\n" + "="*70)
    print("이상치 탐지 알고리즘 비교 결과")
    print("="*70)
    print(f"\n{'알고리즘':<25} {'정밀도':<10} {'재현율':<10} {'F1':<10} {'시간':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['method']:<25} {r['precision']:<10.3f} {r['recall']:<10.3f} "
              f"{r['f1']:<10.3f} {r['time']:.3f}초")

    # 최고 성능 알고리즘
    best_f1 = max(results, key=lambda x: x['f1'])
    best_recall = max(results, key=lambda x: x['recall'])
    fastest = min(results, key=lambda x: x['time'])

    print("\n" + "-"*70)
    print(f"최고 F1: {best_f1['method']} ({best_f1['f1']:.3f})")
    print(f"최고 재현율: {best_recall['method']} ({best_recall['recall']:.3f})")
    print(f"최고 속도: {fastest['method']} ({fastest['time']:.3f}초)")

    return results


def main():
    print("="*70)
    print("이상치 탐지 기법 비교 (4가지 알고리즘)")
    print("="*70)

    results = demo_credit_card_fraud()

    print("\n" + "="*70)
    print("분석 인사이트")
    print("="*70)
    print("""
Isolation Forest:
- 가장 빠른 학습 속도 (O(n log n))
- 대용량 데이터에 적합
- contamination 파라미터로 이상치 비율 설정

One-class SVM:
- 정상 데이터만으로 학습 가능
- RBF 커널로 복잡한 경계 학습
- 대용량 데이터에서 느림 (O(n²~n³))

Local Outlier Factor (LOF):
- 국소 밀도 기반 탐지
- 밀도가 다른 클러스터에서 효과적
- 고차원에서 성능 저하 가능

Autoencoder:
- 딥러닝 기반 비선형 패턴 학습
- 정상 데이터만으로 학습 (One-class)
- 고차원 데이터(이미지, 시계열)에 적합
- 하이퍼파라미터 튜닝 필요

실무 권장:
- 빠른 탐색: Isolation Forest
- 고정밀 탐지: One-class SVM 또는 Autoencoder
- 밀도 기반 분석: LOF
""")


if __name__ == "__main__":
    main()
