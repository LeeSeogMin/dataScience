"""
7장 실습: FT-Transformer Optuna 하이퍼파라미터 튜닝
- 고차원 + 복잡한 상호작용 합성 데이터
- GPU 사용
"""

import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# 1. FT-Transformer 모델
# ============================================================

class FTTransformer(nn.Module):
    """FT-Transformer with configurable hyperparameters"""
    def __init__(self, input_dim, d_model=64, n_heads=4, n_layers=2,
                 dropout=0.1, ff_multiplier=4):
        super().__init__()
        self.d_model = d_model

        # Feature Tokenizer
        self.feature_tokenizer = nn.Sequential(
            nn.Linear(1, d_model),
            nn.LayerNorm(d_model)
        )

        # [CLS] 토큰
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, input_dim + 1, d_model) * 0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        batch_size = x.size(0)

        # Feature tokenization
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        x = self.feature_tokenizer(x)  # (batch, n_features, d_model)

        # Add [CLS] token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1+n_features, d_model)

        # Add positional embedding
        x = x + self.pos_embedding

        # Transformer
        x = self.transformer(x)

        # Use [CLS] output
        cls_output = x[:, 0, :]

        return self.head(cls_output).squeeze(-1)


# ============================================================
# 2. 학습 함수
# ============================================================

def train_and_evaluate(model, train_loader, valid_loader, epochs, lr,
                       weight_decay, device, patience=10):
    """모델 학습 및 검증"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Validation
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

    return best_auc


# ============================================================
# 3. Optuna Objective
# ============================================================

def create_objective(X_train, y_train, X_valid, y_valid, device):
    """Optuna objective function factory"""

    def objective(trial):
        # 하이퍼파라미터 샘플링
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        n_heads = trial.suggest_categorical('n_heads', [2, 4, 8])
        n_layers = trial.suggest_int('n_layers', 1, 4)
        dropout = trial.suggest_float('dropout', 0.0, 0.3)
        ff_multiplier = trial.suggest_categorical('ff_multiplier', [2, 4])
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024])

        # d_model이 n_heads로 나누어지는지 확인
        if d_model % n_heads != 0:
            return 0.5  # Invalid configuration

        # DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        valid_dataset = TensorDataset(
            torch.FloatTensor(X_valid),
            torch.LongTensor(y_valid)
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

        # 모델 생성
        model = FTTransformer(
            input_dim=X_train.shape[1],
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
            ff_multiplier=ff_multiplier
        )

        # 학습 및 평가
        try:
            best_auc = train_and_evaluate(
                model, train_loader, valid_loader,
                epochs=50, lr=lr, weight_decay=weight_decay,
                device=device, patience=10
            )
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.5

        return best_auc

    return objective


# ============================================================
# 4. 메인 실행
# ============================================================

def main():
    # 데이터 로드
    data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(data_dir, 'data', 'synthetic_complex.csv')

    print("=" * 60)
    print("FT-Transformer Optuna 하이퍼파라미터 튜닝")
    print("=" * 60)

    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1).values.astype(np.float32)
    y = df['target'].values.astype(np.int64)

    print(f"데이터: {X.shape[0]}건, {X.shape[1]}특성")

    # 데이터 분할
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
    )

    # 스케일링
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Optuna 튜닝
    print("\n튜닝 시작 (50 trials)...")
    start_time = time.time()

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42)
    )

    objective = create_objective(X_train, y_train, X_valid, y_valid, device)
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    tuning_time = time.time() - start_time

    # 최적 파라미터
    print("\n" + "=" * 60)
    print("튜닝 결과")
    print("=" * 60)
    print(f"최고 Validation AUC: {study.best_value:.4f}")
    print(f"튜닝 시간: {tuning_time:.1f}초 ({tuning_time/60:.1f}분)")
    print("\n최적 하이퍼파라미터:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # 최적 파라미터로 최종 모델 학습 및 테스트
    print("\n최적 파라미터로 최종 평가...")
    best_params = study.best_params

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    valid_dataset = TensorDataset(
        torch.FloatTensor(X_valid),
        torch.LongTensor(y_valid)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.LongTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=best_params['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'])

    # 최종 모델 (더 긴 학습)
    final_model = FTTransformer(
        input_dim=X_train.shape[1],
        d_model=best_params['d_model'],
        n_heads=best_params['n_heads'],
        n_layers=best_params['n_layers'],
        dropout=best_params['dropout'],
        ff_multiplier=best_params['ff_multiplier']
    ).to(device)

    optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    criterion = nn.BCEWithLogitsLoss()

    # 학습
    best_auc = 0
    patience = 20
    patience_counter = 0
    best_state = None

    for epoch in range(100):
        final_model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = final_model(X_batch)
            loss = criterion(outputs, y_batch.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), max_norm=1.0)
            optimizer.step()

        final_model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, _ in valid_loader:
                X_batch = X_batch.to(device)
                outputs = torch.sigmoid(final_model(X_batch))
                val_preds.extend(outputs.cpu().numpy())

        val_auc = roc_auc_score(y_valid, val_preds)
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            # Save best model state (deep copy to preserve tensors)
            best_state = {k: v.clone().cpu() for k, v in final_model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_state is not None:
        final_model.load_state_dict(best_state)
        final_model = final_model.to(device)

    # 테스트 평가
    final_model.eval()
    test_preds = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            outputs = torch.sigmoid(final_model(X_batch))
            test_preds.extend(outputs.cpu().numpy())

    test_pred_binary = (np.array(test_preds) > 0.5).astype(int)
    test_acc = (test_pred_binary == y_test).mean()
    test_auc = roc_auc_score(y_test, test_preds)

    print("\n" + "=" * 60)
    print("최종 테스트 결과 (튜닝된 FT-Transformer)")
    print("=" * 60)
    print(f"정확도: {test_acc:.4f}")
    print(f"AUC: {test_auc:.4f}")

    return study, test_acc, test_auc


if __name__ == "__main__":
    study, acc, auc = main()
