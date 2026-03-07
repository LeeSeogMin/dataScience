"""
7장 실습: 고차원 + 복잡한 상호작용 합성 데이터에서 모델 비교
- XGBoost vs MLP vs TabNet vs FT-Transformer
- 딥러닝 모델이 유리한 조건에서의 성능 검증
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. 합성 데이터 생성
# ============================================================

def generate_complex_synthetic_data(n_samples=50000, n_features=100, n_informative=30,
                                    n_interactions=20, noise_level=0.1, random_state=42):
    """
    고차원 + 복잡한 상호작용이 있는 합성 데이터 생성

    특징:
    - 고차원 (100개 특성)
    - 비선형 상호작용 (곱셈, 제곱, 삼각함수)
    - 정보성 특성과 노이즈 특성 혼합
    """
    np.random.seed(random_state)

    # 기본 특성 생성 (다양한 분포)
    X = np.zeros((n_samples, n_features))

    # 정규분포 특성
    X[:, :n_features//3] = np.random.randn(n_samples, n_features//3)

    # 균등분포 특성
    X[:, n_features//3:2*n_features//3] = np.random.uniform(-2, 2, (n_samples, n_features//3))

    # 지수분포 특성 (스케일링)
    X[:, 2*n_features//3:] = np.random.exponential(1, (n_samples, n_features - 2*n_features//3)) - 1

    # 타겟 변수 생성 (복잡한 비선형 관계)
    y_continuous = np.zeros(n_samples)

    # 1. 선형 효과 (일부 특성만)
    informative_idx = np.random.choice(n_features, n_informative, replace=False)
    weights = np.random.randn(n_informative) * 0.5
    y_continuous += X[:, informative_idx] @ weights

    # 2. 비선형 상호작용 (핵심!)
    for _ in range(n_interactions):
        i, j = np.random.choice(n_informative, 2, replace=False)
        idx_i, idx_j = informative_idx[i], informative_idx[j]
        interaction_type = np.random.choice(['multiply', 'square_sum', 'sin_cos', 'threshold'])

        if interaction_type == 'multiply':
            # 곱셈 상호작용
            y_continuous += 0.3 * X[:, idx_i] * X[:, idx_j]
        elif interaction_type == 'square_sum':
            # 제곱합 상호작용
            y_continuous += 0.2 * (X[:, idx_i]**2 + X[:, idx_j]**2)
        elif interaction_type == 'sin_cos':
            # 삼각함수 상호작용
            y_continuous += 0.4 * np.sin(X[:, idx_i]) * np.cos(X[:, idx_j])
        else:
            # 임계값 기반 상호작용
            y_continuous += 0.5 * ((X[:, idx_i] > 0) & (X[:, idx_j] > 0)).astype(float)

    # 3. 고차 상호작용 (3개 특성)
    for _ in range(n_interactions // 2):
        i, j, k = np.random.choice(n_informative, 3, replace=False)
        idx_i, idx_j, idx_k = informative_idx[i], informative_idx[j], informative_idx[k]
        y_continuous += 0.15 * X[:, idx_i] * X[:, idx_j] * np.sign(X[:, idx_k])

    # 4. 노이즈 추가
    y_continuous += noise_level * np.random.randn(n_samples)

    # 이진 분류로 변환 (중앙값 기준)
    y = (y_continuous > np.median(y_continuous)).astype(int)

    # 특성 이름 생성
    feature_names = [f'feat_{i}' for i in range(n_features)]

    return X.astype(np.float32), y, feature_names, informative_idx

def save_synthetic_data(X, y, feature_names, output_dir):
    """합성 데이터를 CSV로 저장"""
    os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    output_path = os.path.join(output_dir, 'synthetic_complex.csv')
    df.to_csv(output_path, index=False)
    print(f"데이터 저장 완료: {output_path}")
    return output_path

def load_synthetic_data(data_path):
    """저장된 합성 데이터 로드"""
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1).values.astype(np.float32)
    y = df['target'].values.astype(np.int64)
    feature_names = df.columns[:-1].tolist()
    return X, y, feature_names

# ============================================================
# 2. 모델 정의 (이전과 동일)
# ============================================================

class SimpleMLP(nn.Module):
    """다층 퍼셉트론 - 고차원용으로 확장"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class FTTransformer(nn.Module):
    """FT-Transformer - 고차원용으로 조정"""
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.feature_tokenizer = nn.Linear(1, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.unsqueeze(-1)
        x = self.feature_tokenizer(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.transformer(x)
        cls_output = x[:, 0, :]
        return self.head(cls_output).squeeze(-1)


def train_pytorch_model(model, train_loader, valid_loader, epochs=50, lr=1e-3, device='cpu'):
    """PyTorch 모델 학습"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    patience = 15
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            optimizer.step()

        model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for X_batch, y_batch in valid_loader:
                X_batch = X_batch.to(device)
                outputs = torch.sigmoid(model(X_batch))
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(y_batch.numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        scheduler.step(1 - val_auc)

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    return model, best_auc

# ============================================================
# 3. 실험 실행
# ============================================================

def run_comparison_synthetic():
    """합성 데이터에서 4개 모델 성능 비교"""

    # 데이터 경로
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(data_dir, 'data', 'synthetic_complex.csv')

    # 데이터 생성 또는 로드
    if not os.path.exists(data_path):
        print("=" * 60)
        print("합성 데이터 생성 중...")
        print("- 샘플 수: 50,000")
        print("- 특성 수: 100 (정보성 30개)")
        print("- 비선형 상호작용: 20개")
        print("=" * 60)

        X, y, feature_names, informative_idx = generate_complex_synthetic_data(
            n_samples=50000,
            n_features=100,
            n_informative=30,
            n_interactions=20,
            noise_level=0.1
        )
        save_synthetic_data(X, y, feature_names, os.path.join(data_dir, 'data'))
    else:
        print(f"기존 데이터 로드: {data_path}")
        X, y, feature_names = load_synthetic_data(data_path)

    print(f"\n데이터 크기: {X.shape[0]}건, 특성 수: {X.shape[1]}")
    print(f"타겟 분포: 0={np.sum(y==0)}, 1={np.sum(y==1)}")

    # 데이터 분할 (60:20:20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    print(f"학습: {len(X_train)}, 검증: {len(X_valid)}, 테스트: {len(X_test)}")

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    results = {}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    # --- 1. XGBoost ---
    print("=" * 50)
    print("1. XGBoost 학습 중...")
    from xgboost import XGBClassifier

    start_time = time.time()
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=30
    )
    xgb.fit(X_train_scaled, y_train,
            eval_set=[(X_valid_scaled, y_valid)],
            verbose=False)
    xgb_time = time.time() - start_time

    xgb_pred = xgb.predict(X_test_scaled)
    xgb_prob = xgb.predict_proba(X_test_scaled)[:, 1]
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_auc = roc_auc_score(y_test, xgb_prob)

    results['XGBoost'] = {'accuracy': xgb_acc, 'auc': xgb_auc, 'time': xgb_time}
    print(f"   정확도: {xgb_acc:.4f}, AUC: {xgb_auc:.4f}, 시간: {xgb_time:.2f}초")

    # --- 2. MLP ---
    print("\n2. MLP 학습 중...")
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train_scaled),
        torch.LongTensor(y_train)
    )
    valid_dataset = TensorDataset(
        torch.FloatTensor(X_valid_scaled),
        torch.LongTensor(y_valid)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test_scaled),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=512)
    test_loader = DataLoader(test_dataset, batch_size=512)

    start_time = time.time()
    mlp = SimpleMLP(input_dim=X_train.shape[1], hidden_dims=[256, 128, 64])
    mlp, _ = train_pytorch_model(mlp, train_loader, valid_loader, epochs=100, lr=1e-3, device=device)
    mlp_time = time.time() - start_time

    mlp.eval()
    mlp_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = torch.sigmoid(mlp(X_batch))
            mlp_preds.extend(outputs.cpu().numpy())

    mlp_pred = (np.array(mlp_preds) > 0.5).astype(int)
    mlp_acc = accuracy_score(y_test, mlp_pred)
    mlp_auc = roc_auc_score(y_test, mlp_preds)

    results['MLP'] = {'accuracy': mlp_acc, 'auc': mlp_auc, 'time': mlp_time}
    print(f"   정확도: {mlp_acc:.4f}, AUC: {mlp_auc:.4f}, 시간: {mlp_time:.2f}초")

    # --- 3. TabNet ---
    print("\n3. TabNet 학습 중...")
    try:
        from pytorch_tabnet.tab_model import TabNetClassifier

        start_time = time.time()
        tabnet = TabNetClassifier(
            n_d=64, n_a=64,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            verbose=0,
            seed=42
        )
        tabnet.fit(
            X_train_scaled, y_train,
            eval_set=[(X_valid_scaled, y_valid)],
            eval_metric=['auc'],
            max_epochs=100,
            patience=20,
            batch_size=512
        )
        tabnet_time = time.time() - start_time

        tabnet_pred = tabnet.predict(X_test_scaled)
        tabnet_prob = tabnet.predict_proba(X_test_scaled)[:, 1]
        tabnet_acc = accuracy_score(y_test, tabnet_pred)
        tabnet_auc = roc_auc_score(y_test, tabnet_prob)

        results['TabNet'] = {'accuracy': tabnet_acc, 'auc': tabnet_auc, 'time': tabnet_time}
        print(f"   정확도: {tabnet_acc:.4f}, AUC: {tabnet_auc:.4f}, 시간: {tabnet_time:.2f}초")
    except ImportError:
        print("   pytorch-tabnet 미설치. pip install pytorch-tabnet")
        results['TabNet'] = {'accuracy': 0, 'auc': 0, 'time': 0}

    # --- 4. FT-Transformer ---
    print("\n4. FT-Transformer 학습 중...")
    start_time = time.time()
    ft_transformer = FTTransformer(
        input_dim=X_train.shape[1],
        d_model=64,
        n_heads=4,
        n_layers=3,
        dropout=0.1
    )
    ft_transformer, _ = train_pytorch_model(
        ft_transformer, train_loader, valid_loader,
        epochs=100, lr=1e-4, device=device
    )
    ft_time = time.time() - start_time

    ft_transformer.eval()
    ft_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = torch.sigmoid(ft_transformer(X_batch))
            ft_preds.extend(outputs.cpu().numpy())

    ft_pred = (np.array(ft_preds) > 0.5).astype(int)
    ft_acc = accuracy_score(y_test, ft_pred)
    ft_auc = roc_auc_score(y_test, ft_preds)

    results['FT-Transformer'] = {'accuracy': ft_acc, 'auc': ft_auc, 'time': ft_time}
    print(f"   정확도: {ft_acc:.4f}, AUC: {ft_auc:.4f}, 시간: {ft_time:.2f}초")

    # --- 결과 요약 ---
    print("\n" + "=" * 70)
    print("성능 비교 결과 (고차원 + 복잡한 상호작용 합성 데이터)")
    print("=" * 70)
    print(f"{'모델':<18} {'정확도':<10} {'AUC':<10} {'학습 시간':<10}")
    print("-" * 70)
    for model_name, metrics in results.items():
        print(f"{model_name:<18} {metrics['accuracy']:.4f}     {metrics['auc']:.4f}     {metrics['time']:.2f}초")

    # 최고 성능 모델
    best_model = max(results.items(), key=lambda x: x[1]['auc'])
    print(f"\n최고 AUC 모델: {best_model[0]} (AUC: {best_model[1]['auc']:.4f})")

    return results


if __name__ == "__main__":
    results = run_comparison_synthetic()
