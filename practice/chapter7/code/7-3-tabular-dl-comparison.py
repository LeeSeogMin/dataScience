"""
7장 실습: XGBoost vs MLP vs TabNet vs FT-Transformer 성능 비교
- Adult Income 데이터셋 사용
- 동일한 전처리 조건에서 4개 모델 비교
"""

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. 데이터 준비
# ============================================================

def load_adult_data():
    """Adult Income 데이터셋 로드 및 전처리"""
    from sklearn.datasets import fetch_openml

    print("Adult Income 데이터셋 로딩 중...")
    adult = fetch_openml(name='adult', version=2, as_frame=True)
    df = adult.frame

    # 타겟 변수 이진화 (컬럼명: 'class')
    df['target'] = (df['class'] == '>50K').astype(int)

    # 결측치 처리
    df = df.replace('?', np.nan).dropna()

    # 특성과 타겟 분리
    X = df.drop(['class', 'target'], axis=1)
    y = df['target'].values

    # 범주형/수치형 분리
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # 범주형 인코딩
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    X = X.values.astype(np.float32)
    y = y.astype(np.int64)

    print(f"데이터 크기: {X.shape[0]}건, 특성 수: {X.shape[1]}")
    return X, y, cat_cols, num_cols

# ============================================================
# 2. 모델 정의
# ============================================================

class SimpleMLP(nn.Module):
    """단순 다층 퍼셉트론"""
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.2):
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
    """간소화된 FT-Transformer 구현"""
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Feature Tokenizer: 각 특성을 d_model 차원으로 변환
        self.feature_tokenizer = nn.Linear(1, d_model)

        # [CLS] 토큰
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # 출력 헤드
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 각 특성을 토큰으로 변환: (batch, n_features) -> (batch, n_features, d_model)
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        x = self.feature_tokenizer(x)  # (batch, n_features, d_model)

        # [CLS] 토큰 추가
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+n_features, d_model)

        # Transformer
        x = self.transformer(x)

        # [CLS] 출력만 사용
        cls_output = x[:, 0, :]

        return self.head(cls_output).squeeze(-1)


def train_pytorch_model(model, train_loader, valid_loader, epochs=50, lr=1e-3, device='cpu'):
    """PyTorch 모델 학습"""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    patience = 10
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

        # 검증
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

def run_comparison():
    """4개 모델 성능 비교"""
    # 데이터 로드
    X, y, cat_cols, num_cols = load_adult_data()

    # 데이터 분할 (60:20:20)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

    print(f"\n학습: {len(X_train)}, 검증: {len(X_valid)}, 테스트: {len(X_test)}")

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
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=20
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

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=256)
    test_loader = DataLoader(test_dataset, batch_size=256)

    start_time = time.time()
    mlp = SimpleMLP(input_dim=X_train.shape[1], hidden_dims=[128, 64])
    mlp, _ = train_pytorch_model(mlp, train_loader, valid_loader, epochs=100, device=device)
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
            n_d=32, n_a=32,
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
            batch_size=256
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
        n_layers=2,
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
    print("\n" + "=" * 60)
    print("성능 비교 결과 (Adult Income 데이터셋)")
    print("=" * 60)
    print(f"{'모델':<18} {'정확도':<10} {'AUC':<10} {'학습 시간':<10}")
    print("-" * 60)
    for model_name, metrics in results.items():
        print(f"{model_name:<18} {metrics['accuracy']:.4f}     {metrics['auc']:.4f}     {metrics['time']:.2f}초")

    return results


if __name__ == "__main__":
    results = run_comparison()
