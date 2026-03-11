"""
7장 실습: PyTorch 기반 실무 파이프라인
- 재현성 확보
- DataLoader 구성
- 정규화 기법 (Dropout, BatchNorm, Weight Decay)
- 조기 중단과 모델 저장
- 전체 학습 루프
"""

import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import os

# ============================================================
# 1. 재현성 확보
# ============================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

print("=" * 60)
print("PyTorch 기반 실무 파이프라인")
print("=" * 60)

# ============================================================
# 2. 데이터 로드 및 전처리
# ============================================================

print("\n1. 데이터 로드...")

# 합성 데이터 로드 (7.3절에서 생성한 데이터)
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(os.path.dirname(script_dir), 'data', 'synthetic_complex.csv')

df = pd.read_csv(data_path)
X = df.drop('target', axis=1).values.astype(np.float32)
y = df['target'].values.astype(np.int64)

print(f"   데이터: {X.shape[0]}건, {X.shape[1]}특성")
print(f"   클래스 비율: {y.mean():.1%} positive")

# 데이터 분할
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=0.2, random_state=SEED, stratify=y_temp
)

# 스케일링
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

print(f"   학습: {len(X_train)}, 검증: {len(X_valid)}, 테스트: {len(X_test)}")

# ============================================================
# 3. DataLoader 구성
# ============================================================

print("\n2. DataLoader 구성...")

batch_size = 256
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"   Device: {device}")
print(f"   배치 크기: {batch_size}")

# 텐서 변환
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
X_valid_t = torch.FloatTensor(X_valid)
y_valid_t = torch.LongTensor(y_valid)
X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)

# DataLoader
train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)
valid_loader = DataLoader(
    TensorDataset(X_valid_t, y_valid_t),
    batch_size=batch_size
)
test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t),
    batch_size=batch_size
)

# ============================================================
# 4. 모델 정의 (정규화 기법 포함)
# ============================================================

print("\n3. 모델 정의...")

class MLP(nn.Module):
    """정규화 기법이 적용된 MLP"""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

model = MLP(input_dim=X_train.shape[1]).to(device)
print(f"   파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

# ============================================================
# 5. 조기 중단 클래스
# ============================================================

class EarlyStopping:
    """조기 중단 구현"""
    def __init__(self, patience=10):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.best_state = None

    def update(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            # 깊은 복사로 모델 상태 저장
            self.best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            self.counter = 0
            return True  # 개선됨
        else:
            self.counter += 1
            return False  # 개선 안됨

    @property
    def should_stop(self):
        return self.counter >= self.patience

# ============================================================
# 6. 학습 설정
# ============================================================

print("\n4. 학습 설정...")

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)
early_stopping = EarlyStopping(patience=15)

max_epochs = 100
print(f"   최대 에폭: {max_epochs}")
print(f"   조기 중단 patience: 15")
print(f"   Weight Decay: 1e-5")

# ============================================================
# 7. 학습 루프
# ============================================================

print("\n5. 학습 시작...")

history = {'train_loss': [], 'valid_loss': [], 'valid_auc': []}

for epoch in range(max_epochs):
    # 학습
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.float())
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # 검증
    model.eval()
    valid_loss = 0
    valid_preds = []
    valid_targets = []

    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.float())
            valid_loss += loss.item()

            probs = torch.sigmoid(outputs)
            valid_preds.extend(probs.cpu().numpy())
            valid_targets.extend(y_batch.cpu().numpy())

    valid_loss /= len(valid_loader)
    valid_auc = roc_auc_score(valid_targets, valid_preds)

    # 기록
    history['train_loss'].append(train_loss)
    history['valid_loss'].append(valid_loss)
    history['valid_auc'].append(valid_auc)

    # 학습률 스케줄링
    scheduler.step(valid_loss)

    # 조기 중단 확인
    improved = early_stopping.update(valid_loss, model)

    if epoch % 10 == 0 or improved:
        status = "✓" if improved else ""
        print(f"   Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
              f"valid_loss={valid_loss:.4f}, valid_auc={valid_auc:.4f} {status}")

    if early_stopping.should_stop:
        print(f"\n   조기 중단 (Epoch {epoch+1})")
        break

# 최적 모델 복원
if early_stopping.best_state is not None:
    model.load_state_dict(early_stopping.best_state)
    model = model.to(device)

# ============================================================
# 8. 테스트 평가
# ============================================================

print("\n6. 테스트 평가...")

model.eval()
test_preds = []
test_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        probs = torch.sigmoid(outputs)
        test_preds.extend(probs.cpu().numpy())
        test_targets.extend(y_batch.numpy())

test_preds = np.array(test_preds)
test_pred_binary = (test_preds > 0.5).astype(int)

test_acc = accuracy_score(test_targets, test_pred_binary)
test_auc = roc_auc_score(test_targets, test_preds)

# ============================================================
# 9. 결과 출력
# ============================================================

print("\n" + "=" * 60)
print("결과 요약")
print("=" * 60)
print(f"학습 완료 에폭: {len(history['train_loss'])}")
print(f"최적 검증 손실: {early_stopping.best_loss:.4f}")
print(f"최종 검증 AUC: {history['valid_auc'][-1]:.4f}")
print(f"\n테스트 결과:")
print(f"  정확도: {test_acc:.4f}")
print(f"  AUC: {test_auc:.4f}")

# 결과 저장
output_dir = os.path.join(os.path.dirname(script_dir), 'data', 'output')
os.makedirs(output_dir, exist_ok=True)

results = {
    'epochs': len(history['train_loss']),
    'best_valid_loss': early_stopping.best_loss,
    'test_accuracy': test_acc,
    'test_auc': test_auc
}

pd.DataFrame([results]).to_csv(
    os.path.join(output_dir, 'pytorch_pipeline_results.csv'),
    index=False
)
print(f"\n결과 저장: {output_dir}/pytorch_pipeline_results.csv")
